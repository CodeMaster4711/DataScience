"""
Dataset loader for databricks-dolly-15k
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from typing import Dict, List
from config import TransformerConfig


class DollyDataset(Dataset):
    """
    Databricks Dolly 15k Dataset for instruction following
    Format: {"instruction": "...", "context": "...", "response": "..."}
    """
    def __init__(self, config: TransformerConfig, split: str = "train"):
        """
        Args:
            config: TransformerConfig object
            split: "train" or "validation"
        """
        self.config = config
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # GPT-2 doesn't have a pad token by default
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading databricks-dolly-15k dataset ({split} split)...")

        # Load dataset from HuggingFace - Use configurable subset
        dataset = load_dataset("databricks/databricks-dolly-15k", split=f"train[:{config.dataset_size}]")
        print(f"Using subset of {config.dataset_size} samples (instead of 15k) for improved training")

        # Split into train/val
        if split == "train":
            total_samples = len(dataset)
            train_size = int(config.train_split * total_samples)
            dataset = dataset.select(range(train_size))
        elif split == "validation":
            total_samples = len(dataset)
            train_size = int(config.train_split * total_samples)
            dataset = dataset.select(range(train_size, total_samples))

        print(f"Loaded {len(dataset)} samples for {split}")

        # Preprocess all samples
        self.samples = []
        for item in dataset:
            processed = self._preprocess_item(item)
            if processed is not None:
                self.samples.append(processed)

        print(f"Preprocessed {len(self.samples)} samples")

    def _preprocess_item(self, item: Dict) -> Dict:
        """
        Preprocess a single item from the dataset
        Format the instruction + context + response into a single text
        """
        instruction = item.get('instruction', '').strip()
        context = item.get('context', '').strip()
        response = item.get('response', '').strip()

        # Skip empty samples
        if not instruction or not response:
            return None

        # Format: "Instruction: {instruction}\n{context}\nResponse: {response}"
        if context:
            text = f"Instruction: {instruction}\nContext: {context}\nResponse: {response}"
        else:
            text = f"Instruction: {instruction}\nResponse: {response}"

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        # Skip if too long
        if len(tokens) > self.config.max_seq_length:
            tokens = tokens[:self.config.max_seq_length]

        return {
            'tokens': tokens,
            'length': len(tokens),
            'text': text
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a tokenized sample
        """
        sample = self.samples[idx]
        tokens = sample['tokens']

        # Create input and target sequences
        # Input: tokens[:-1], Target: tokens[1:]
        # This is for autoregressive training: predict next token
        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'length': len(input_ids)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to pad sequences to the same length
    """
    # Find max length in batch
    max_length = max(item['length'] for item in batch)

    input_ids = []
    target_ids = []
    attention_mask = []

    for item in batch:
        length = item['length']
        pad_length = max_length - length

        # Pad input_ids
        padded_input = torch.cat([
            item['input_ids'],
            torch.full((pad_length,), fill_value=50256, dtype=torch.long)  # GPT-2 eos_token_id
        ])
        input_ids.append(padded_input)

        # Pad target_ids with -100 (ignore index for loss)
        padded_target = torch.cat([
            item['target_ids'],
            torch.full((pad_length,), fill_value=-100, dtype=torch.long)
        ])
        target_ids.append(padded_target)

        # Create attention mask (1 for real tokens, 0 for padding)
        mask = torch.cat([
            torch.ones(length, dtype=torch.long),
            torch.zeros(pad_length, dtype=torch.long)
        ])
        attention_mask.append(mask)

    return {
        'input_ids': torch.stack(input_ids),
        'target_ids': torch.stack(target_ids),
        'attention_mask': torch.stack(attention_mask)
    }


def create_dataloaders(config: TransformerConfig):
    """
    Create train and validation dataloaders
    """
    train_dataset = DollyDataset(config, split="train")
    val_dataset = DollyDataset(config, split="validation")

    # Optimize num_workers for Mac vs Linux/Windows
    # Use 0 workers on Mac for improved model to save memory
    num_train_workers = 0 if config.device == "mps" else 4  # Mac: single process
    num_val_workers = 0 if config.device == "mps" else 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_train_workers,
        pin_memory=True if config.device == "cuda" else False,  # Only for CUDA
        persistent_workers=True if num_train_workers > 0 else False,
        prefetch_factor=2 if num_train_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_val_workers,
        pin_memory=True if config.device == "cuda" else False,  # Only for CUDA
        persistent_workers=True if num_val_workers > 0 else False,
        prefetch_factor=2 if num_val_workers > 0 else None
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    config = TransformerConfig()
    train_loader, val_loader = create_dataloaders(config)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Show a sample batch
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Target shape: {batch['target_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
