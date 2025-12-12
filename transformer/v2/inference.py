"""
Inference script for Transformer LLM - Text Generation
"""
import torch
from transformers import GPT2Tokenizer
from pathlib import Path
import argparse

from config import TransformerConfig
from model import TransformerLLM


class TextGenerator:
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Initialize text generator with a trained model

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on ('cuda', 'mps', or 'cpu')
        """
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Get config from checkpoint
        self.config = checkpoint['config']

        # Set device
        if device is None:
            device = self.config.device
        self.device = device
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = TransformerLLM(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")

        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = None,
        num_return_sequences: int = 1
    ) -> list[str]:
        """
        Generate text from a prompt

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (None = no filtering)
            top_p: Nucleus sampling (None = no filtering)
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generated texts
        """
        self.model.eval()

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        # Generate multiple sequences
        generated_texts = []
        for _ in range(num_return_sequences):
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )

            # Decode
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_texts.append(generated_text)

        return generated_texts

    def chat(self):
        """Interactive chat mode"""
        print("\n" + "="*80)
        print("Interactive Chat Mode")
        print("Type 'quit' or 'exit' to stop")
        print("="*80 + "\n")

        while True:
            # Get user input
            prompt = input("\nYou: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not prompt:
                continue

            # Format as instruction
            formatted_prompt = f"Instruction: {prompt}\nResponse:"

            # Generate response
            print("\nGenerating...", end='\r')
            generated = self.generate(
                formatted_prompt,
                max_new_tokens=150,
                temperature=0.7,
                top_k=50
            )[0]

            # Extract only the response part
            if "Response:" in generated:
                response = generated.split("Response:")[-1].strip()
            else:
                response = generated

            print(f"\nAssistant: {response}")


def main():
    parser = argparse.ArgumentParser(description="Generate text with Transformer LLM")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Text prompt for generation'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=100,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='Top-k sampling'
    )
    parser.add_argument(
        '--chat',
        action='store_true',
        help='Start interactive chat mode'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/mps/cpu)'
    )

    args = parser.parse_args()

    # Initialize generator
    generator = TextGenerator(args.checkpoint, device=args.device)

    if args.chat:
        # Interactive mode
        generator.chat()
    elif args.prompt:
        # Single generation
        print(f"\nPrompt: {args.prompt}\n")

        generated = generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )[0]

        print(f"Generated:\n{generated}\n")
    else:
        # Demo mode with example prompts
        example_prompts = [
            "Instruction: What is machine learning?\nResponse:",
            "Instruction: Explain the concept of neural networks.\nResponse:",
            "Instruction: How do transformers work?\nResponse:",
            "Instruction: Write a Python function to calculate factorial.\nResponse:"
        ]

        print("\n" + "="*80)
        print("Demo Mode - Generating responses for example prompts")
        print("="*80)

        for prompt in example_prompts:
            print(f"\n{'-'*80}")
            print(f"Prompt: {prompt}")
            print(f"{'-'*80}")

            generated = generator.generate(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )[0]

            print(f"\nGenerated:\n{generated}\n")


if __name__ == "__main__":
    main()
