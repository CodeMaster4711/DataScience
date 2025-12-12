"""
Configuration for Transformer LLM Training - v2 IMPROVED
Better hyperparameters for improved text quality while maintaining Mac M3 compatibility
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
    # Model Architecture - SMALL for Mac M3 (~15M parameters - improved from 7M)
    vocab_size: int = 50257  # GPT-2 tokenizer vocabulary size
    d_model: int = 192  # Embedding dimension (improved from 128)
    n_heads: int = 6  # Number of attention heads (improved from 4)
    n_layers: int = 4  # Number of transformer blocks (improved from 3)
    d_ff: int = 768  # Feed-forward dimension (4x d_model)
    max_seq_length: int = 128  # Maximum sequence length
    dropout: float = 0.1  # Dropout rate

    # Attention Configuration
    attention_dropout: float = 0.1
    use_bias: bool = True  # Use bias in linear layers

    # Training Hyperparameters - IMPROVED for v2
    batch_size: int = 1  # Small batch for memory efficiency
    learning_rate: float = 2e-4  # Slightly lower for better convergence
    weight_decay: float = 0.01
    epochs: int = 25  # Increased from 10 for better learning
    warmup_steps: int = 1000  # Increased for smoother learning
    max_grad_norm: float = 1.0  # Gradient clipping

    # Performance Optimization
    use_amp: bool = False  # Not supported on MPS
    gradient_accumulation_steps: int = 16  # Maintain effective batch_size=16
    compile_model: bool = False  # Can cause issues on Mac

    # Dataset Configuration - MORE DATA for better learning
    train_split: float = 0.98
    val_split: float = 0.02
    dataset_size: int = 10000  # Increased from 5000 (use first 10k samples)

    # Device Configuration
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Logging & Checkpointing - IMPROVED monitoring
    log_interval: int = 50  # Log every N steps
    eval_interval: int = 500  # More frequent evaluation (was 1000)
    save_interval: int = 1000  # More frequent checkpoints (was 2000)
    checkpoint_dir: str = "checkpoints"

    # wandb Configuration (can be overridden by .env)
    use_wandb: bool = True
    wandb_project: str = "transformer-llm-v2"  # Separate project for v2
    wandb_entity: str = "codemaster4711"

    def __post_init__(self):
        # Override with environment variables if they exist
        if os.getenv('WANDB_PROJECT'):
            self.wandb_project = os.getenv('WANDB_PROJECT')
        if os.getenv('WANDB_ENTITY'):
            self.wandb_entity = os.getenv('WANDB_ENTITY')

        # Platform-specific optimizations
        if self.device == "mps":
            print(f"Mac M3 (MPS) detected - Using SMALL-v2 configuration:")
            print(f"  - Model: {self.d_model}d × {self.n_layers} layers = ~15M parameters")
            print(f"  - Batch size: {self.batch_size}, Gradient accumulation: {self.gradient_accumulation_steps}")
            print(f"  - Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
            print(f"  - Sequence length: {self.max_seq_length}")
            print(f"  - Epochs: {self.epochs}")
            print(f"  - Dataset size: {self.dataset_size} samples")

            # Set MPS memory fraction to leave room for OS
            try:
                torch.mps.set_per_process_memory_fraction(0.7)  # Use max 70% of available memory
                print(f"  - MPS memory fraction: 0.7 (70% of available memory)")
            except:
                pass

        # Validate configuration
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.train_split + self.val_split <= 1.0, "train_split + val_split must be <= 1.0"
