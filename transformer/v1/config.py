"""
Configuration for Transformer LLM Training
"""
from dataclasses import dataclass
import torch
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from transformer directory
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)


@dataclass
class TransformerConfig:
    # Model Architecture - TINY for Mac M3 (7M parameters)
    vocab_size: int = 50257  # GPT-2 tokenizer vocabulary size
    d_model: int = 128  # Embedding dimension (reduced from 512)
    n_heads: int = 4  # Number of attention heads (reduced from 8)
    n_layers: int = 3  # Number of transformer blocks (reduced from 6)
    d_ff: int = 512  # Feed-forward dimension (4x d_model, reduced from 2048)
    max_seq_length: int = 128  # Maximum sequence length (reduced from 512)
    dropout: float = 0.1  # Dropout rate

    # Attention Configuration
    attention_dropout: float = 0.1
    use_bias: bool = True  # Use bias in linear layers

    # Training Hyperparameters - TINY for Mac M3
    batch_size: int = 1  # Small batch for memory efficiency
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_steps: int = 500  # Reduced from 1000 for smaller model
    max_grad_norm: float = 1.0  # Gradient clipping

    # Performance Optimization
    use_amp: bool = False  # Not supported on MPS
    gradient_accumulation_steps: int = 16  # Increased to maintain effective batch_size=16
    compile_model: bool = False  # Can cause issues on Mac

    # Dataset Configuration
    train_split: float = 0.98  # More training data, less validation
    val_split: float = 0.02

    # Device Configuration
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Logging & Checkpointing
    log_interval: int = 50  # Log every N steps (reduced logging overhead)
    eval_interval: int = 1000  # Evaluate every N steps (less frequent evaluation)
    save_interval: int = 2000  # Save checkpoint every N steps (less frequent saving)
    checkpoint_dir: str = "checkpoints"

    # wandb Configuration (can be overridden by .env)
    use_wandb: bool = True
    wandb_project: str = "transformer-llm"
    wandb_entity: str = "codemaster4711"

    def __post_init__(self):
        # Override with environment variables if they exist
        if os.getenv('WANDB_PROJECT'):
            self.wandb_project = os.getenv('WANDB_PROJECT')
        if os.getenv('WANDB_ENTITY'):
            self.wandb_entity = os.getenv('WANDB_ENTITY')

        # Platform-specific optimizations
        if self.device == "mps":
            print(f"Mac M3 (MPS) detected - Using TINY configuration:")
            print(f"  - Model: {self.d_model}d × {self.n_layers} layers = ~7M parameters")
            print(f"  - Batch size: {self.batch_size}, Gradient accumulation: {self.gradient_accumulation_steps}")
            print(f"  - Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
            print(f"  - Sequence length: {self.max_seq_length}")

            # Set MPS memory fraction to leave room for OS
            try:
                torch.mps.set_per_process_memory_fraction(0.7)  # Use max 70% of available memory
                print(f"  - MPS memory fraction: 0.7 (70% of available memory)")
            except:
                pass

        # Validate configuration
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.train_split + self.val_split <= 1.0, "train_split + val_split must be <= 1.0"
