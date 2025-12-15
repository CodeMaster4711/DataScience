"""
Gradio Web UI for TINY Transformer LLM
Interactive interface for testing text generation
"""
import gradio as gr
import torch
from pathlib import Path
from transformers import GPT2Tokenizer
import sys

from config import TransformerConfig
from model import TransformerLLM


class GradioInference:
    def __init__(self, checkpoint_path: str = "checkpoints/best_model.pt"):
        """
        Initialize the inference model with a trained checkpoint

        Args:
            checkpoint_path: Path to the model checkpoint
        """
        print("Loading model...")
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                f"Please train the model first or specify a valid checkpoint."
            )

        # Load checkpoint (weights_only=False for PyTorch 2.6+ compatibility)
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        self.config = checkpoint['config']

        # Set device
        self.device = self.config.device
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
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40
    ) -> str:
        """
        Generate text from a prompt

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling

        Returns:
            Generated text
        """
        if not prompt.strip():
            return "Please provide a prompt!"

        self.model.eval()

        # Format prompt as instruction if not already formatted
        if "Instruction:" not in prompt and "Response:" not in prompt:
            formatted_prompt = f"Instruction: {prompt}\nResponse:"
        else:
            formatted_prompt = prompt

        # Tokenize
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors='pt').to(self.device)

        # Generate
        try:
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )

            # Decode
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            return generated_text

        except Exception as e:
            return f"Error during generation: {str(e)}"


def create_gradio_interface(checkpoint_path: str = "checkpoints/best_model.pt"):
    """
    Create and launch Gradio interface

    Args:
        checkpoint_path: Path to model checkpoint
    """
    # Initialize inference
    try:
        inference = GradioInference(checkpoint_path)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train the model first by running:")
        print("  python train.py")
        sys.exit(1)

    # Create Gradio interface
    with gr.Blocks(title="TINY Transformer LLM") as demo:
        gr.Markdown(
            """
            # 🤖 TINY Transformer LLM Demo

            Interactive text generation with your trained transformer model.

            **Model Details:**
            - Parameters: ~7M
            - Architecture: 3-layer Transformer (128d)
            - Trained on: databricks-dolly-15k (subset)
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")

                # Input prompt
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your instruction or question here...",
                    lines=4,
                    value="What is machine learning?"
                )

                # Generation parameters
                with gr.Accordion("Generation Parameters", open=False):
                    max_tokens = gr.Slider(
                        minimum=10,
                        maximum=128,
                        value=50,
                        step=10,
                        label="Max New Tokens",
                        info="Maximum number of tokens to generate"
                    )

                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="Temperature",
                        info="Higher = more creative, Lower = more focused"
                    )

                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=40,
                        step=1,
                        label="Top-k",
                        info="Number of top tokens to consider"
                    )

                # Generate button
                generate_btn = gr.Button("Generate", variant="primary", size="lg")

                # Example prompts
                gr.Examples(
                    examples=[
                        ["What is machine learning?"],
                        ["Explain Python programming in simple terms."],
                        ["How do neural networks work?"],
                        ["What is the difference between AI and ML?"],
                        ["Write a Python function to calculate factorial."],
                    ],
                    inputs=prompt_input,
                    label="Example Prompts"
                )

            with gr.Column(scale=1):
                gr.Markdown("### Output")

                # Output
                output_text = gr.Textbox(
                    label="Generated Text",
                    lines=15
                )

                # Model info
                gr.Markdown(
                    f"""
                    **Model Info:**
                    - Checkpoint: `{checkpoint_path}`
                    - Device: {inference.device}
                    - Parameters: {inference.model.count_parameters():,}
                    - Sequence Length: {inference.config.max_seq_length}
                    """
                )

        # Connect generate button
        generate_btn.click(
            fn=inference.generate,
            inputs=[prompt_input, max_tokens, temperature, top_k],
            outputs=output_text
        )

        # Footer
        gr.Markdown(
            """
            ---
            Built with Gradio • Powered by PyTorch • TINY Transformer LLM
            """
        )

    return demo


def main():
    """Main function to launch Gradio app"""
    import argparse

    parser = argparse.ArgumentParser(description="Launch Gradio UI for TINY Transformer LLM")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint (default: checkpoints/best_model.pt)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public share link'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run the server on (default: 7860)'
    )

    args = parser.parse_args()

    # Create and launch interface
    demo = create_gradio_interface(args.checkpoint)

    print("\n" + "="*80)
    print("Launching Gradio Interface...")
    print("="*80 + "\n")

    demo.launch(
        share=args.share,
        server_port=args.port,
        show_error=True,
        inbrowser=True  # Auto-open in browser
    )


if __name__ == "__main__":
    main()
