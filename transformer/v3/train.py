"""
Training script for Transformer LLM with wandb integration
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import os
from pathlib import Path
from tqdm import tqdm
import math
from dotenv import load_dotenv
from contextlib import nullcontext

from config import TransformerConfig
from model import TransformerLLM
from dataset import create_dataloaders
from transformers import GPT2Tokenizer


class Trainer:
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.device = config.device
        print(f"Using device: {self.device}")

        # Initialize model
        print("Initializing model...")
        self.model = TransformerLLM(config).to(self.device)
        print(f"Model parameters: {self.model.count_parameters():,}")

        # Enable gradient checkpointing only if explicitly enabled
        if config.use_gradient_checkpointing:
            print("Enabling gradient checkpointing for Mac M3 memory savings (~40% reduction)")
            self.model.enable_gradient_checkpointing()
        else:
            print("Gradient checkpointing disabled (config.use_gradient_checkpointing=False)")

        # Clear MPS cache if using Mac
        if config.device == "mps":
            print("Clearing MPS memory cache...")
            torch.mps.empty_cache()

        # Compile model for PyTorch 2.0+ (20-30% speedup)
        if config.compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile() for optimization...")
            try:
                self.model = torch.compile(self.model)
                print("Model compiled successfully!")
            except Exception as e:
                print(f"Warning: Could not compile model: {e}")
                print("Continuing without compilation...")

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )

        # Learning rate scheduler - v3 CRITICAL FIX
        # Use configurable eta_min to prevent LR from dying to 0
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.scheduler_eta_min  # v3: 1e-5 instead of 1e-6 (LR doesn't die!)
        )

        # Create dataloaders
        print("Loading dataset...")
        self.train_loader, self.val_loader = create_dataloaders(config)

        # Tokenizer for text generation
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Mixed Precision Training (AMP) - 2x speedup
        self.scaler = torch.cuda.amp.GradScaler() if (config.use_amp and config.device == "cuda") else None
        if self.scaler:
            print("Using Automatic Mixed Precision (AMP) training for 2x speedup")

        # Initialize wandb
        if config.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases"""
        # Load .env from transformer directory (parent of v1)
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(env_path)

        wandb_api_key = os.getenv('WANDB_API_KEY')
        wandb_base_url = os.getenv('WANDB_BASE_URL')

        if wandb_api_key:
            wandb.login(key=wandb_api_key)

        # Set base URL for local wandb server if provided
        if wandb_base_url:
            os.environ['WANDB_BASE_URL'] = wandb_base_url

        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            config={
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "n_layers": self.config.n_layers,
                "d_ff": self.config.d_ff,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "max_seq_length": self.config.max_seq_length,
                "vocab_size": self.config.vocab_size,
                "dropout": self.config.dropout,
                "total_parameters": self.model.count_parameters(),
            }
        )
        wandb.watch(self.model, log="all", log_freq=100)

    def train_epoch(self, epoch):
        """Train for one epoch with gradient accumulation and mixed precision"""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        accumulation_counter = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            # Mixed Precision Training with autocast
            with torch.cuda.amp.autocast() if self.scaler else nullcontext():
                # Forward pass
                logits, loss = self.model(input_ids, target_ids)

                # CRITICAL: Save unscaled loss for metrics BEFORE scaling!
                unscaled_loss = loss.item()

                # Scale loss for gradient accumulation (ONLY for backward pass!)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass with gradient scaling
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_counter += 1

            # Update weights only after accumulating gradients
            if accumulation_counter >= self.config.gradient_accumulation_steps:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                accumulation_counter = 0

            # Calculate perplexity using UNSCALED loss (the REAL loss!)
            perplexity = math.exp(unscaled_loss)

            # Track metrics using UNSCALED loss (the REAL loss!)
            total_loss += unscaled_loss
            num_tokens = (target_ids != -100).sum().item()
            total_tokens += num_tokens

            # Update progress bar (using UNSCALED loss for display!)
            pbar.set_postfix({
                'loss': f'{unscaled_loss:.4f}',
                'ppl': f'{perplexity:.2f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'acc': f'{accumulation_counter}/{self.config.gradient_accumulation_steps}'
            })

            # Increment global step only after actual optimizer step
            if accumulation_counter == 0:
                self.global_step += 1

            # Log to wandb (only after optimizer step, using UNSCALED loss!)
            if accumulation_counter == 0 and self.config.use_wandb and self.global_step % self.config.log_interval == 0:
                wandb.log({
                    'train/loss': unscaled_loss,
                    'train/perplexity': perplexity,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/tokens': total_tokens,
                    'epoch': epoch,
                    'step': self.global_step
                })

            # Validation (only after optimizer step)
            if accumulation_counter == 0 and self.global_step > 0 and self.global_step % self.config.eval_interval == 0:
                val_loss, val_ppl = self.validate()
                print(f"\nValidation - Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")

                if self.config.use_wandb:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/perplexity': val_ppl,
                        'step': self.global_step
                    })

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    print(f"Saved best model (val_loss: {val_loss:.4f})")

                # Generate sample text
                self.generate_sample()

                # Clear MPS cache to prevent memory buildup
                if self.config.device == "mps":
                    torch.mps.empty_cache()

                self.model.train()

            # Save checkpoint (only after optimizer step)
            if accumulation_counter == 0 and self.global_step > 0 and self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            logits, loss = self.model(input_ids, target_ids)

            total_loss += loss.item()
            num_tokens = (target_ids != -100).sum().item()
            total_tokens += num_tokens

        avg_loss = total_loss / len(self.val_loader)
        perplexity = math.exp(avg_loss)

        return avg_loss, perplexity

    @torch.no_grad()
    def generate_sample(self):
        """Generate sample text to monitor training progress"""
        self.model.eval()

        prompts = [
            "Instruction: What is machine learning?\nResponse:",
            "Instruction: Explain Python programming.\nResponse:",
            "Instruction: How do neural networks work?\nResponse:"
        ]

        for prompt in prompts:
            # Tokenize prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

            # Generate
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_k=40
            )

            # Decode
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print(f"\n{'='*80}")
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            print(f"{'='*80}\n")

            if self.config.use_wandb:
                wandb.log({
                    f'samples/generation': wandb.Html(f"<pre>{generated_text}</pre>"),
                    'step': self.global_step
                })

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from {filename}")

    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config.epochs} epochs...")
        print(f"Total steps: {len(self.train_loader) * self.config.epochs}")

        for epoch in range(1, self.config.epochs + 1):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{self.config.epochs}")
            print(f"{'='*80}")

            train_loss = self.train_epoch(epoch)
            val_loss, val_ppl = self.validate()

            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Perplexity: {val_ppl:.2f}")

            # Step scheduler
            self.scheduler.step()

            # Save epoch checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        print("\nTraining completed!")
        if self.config.use_wandb:
            wandb.finish()


def main():
    # Load configuration
    config = TransformerConfig()

    # Create trainer
    trainer = Trainer(config)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
