# Transformer LLM v2 - Improved Quality Model

This is v2 of the TINY Transformer LLM with **significant improvements** over v1 for better text generation quality while remaining Mac M3 compatible.

## 🚀 Key Improvements Over v1

### Model Architecture (7M → 15M parameters)
- **d_model**: 128 → **192** (50% increase in embedding dimensions)
- **n_heads**: 4 → **6** (more attention perspectives)
- **n_layers**: 3 → **4** (deeper model)
- **d_ff**: 512 → **768** (larger feed-forward capacity)

### Training Configuration
- **epochs**: 10 → **25** (2.5x more training)
- **dataset_size**: 5,000 → **10,000** samples (2x more data)
- **learning_rate**: 3e-4 → **2e-4** (more stable convergence)
- **warmup_steps**: 500 → **1,000** (better learning rate warmup)

### Better Monitoring
- **eval_interval**: 1,000 → **500** steps (more frequent validation)
- **save_interval**: 2,000 → **1,000** steps (more checkpoints)
- **log_interval**: 50 → **25** steps (detailed logging)

## 📊 Expected Results

### v1 Problems (Why v2 was needed)
- ❌ Validation loss: ~6.62 (too high)
- ❌ Output quality: Repetitive nonsense
- ❌ Training: Only 1,000 steps
- ❌ Model capacity: Too small (7M params)

Example v1 output:
```
"percussion or is the the how is the percussion or a best the The the Marvel..."
```

### v2 Expected Improvements
- ✅ Target validation loss: <3.0 (much better)
- ✅ Output quality: Coherent sentences
- ✅ Training: ~15,625 steps (15.6x more)
- ✅ Model capacity: 15M parameters (2.14x larger)

## 🏗️ Architecture Details

```
TransformerLLM(
  (token_embedding): Embedding(50257, 192)
  (positional_encoding): PositionalEncoding(dropout=0.1)
  (transformer_blocks): ModuleList(
    (0-3): 4 x TransformerBlock(
      (attention): MultiHeadAttention(
        n_heads=6, d_k=32, d_model=192
      )
      (feed_forward): FeedForward(192 -> 768 -> 192)
      (ln1, ln2): LayerNorm(192)
    )
  )
  (ln_final): LayerNorm(192)
  (output_projection): Linear(192 -> 50257)
)
```

**Total Parameters**: ~15,000,000 (15M)

## 🖥️ Mac M3 Optimizations (Maintained from v1)

Despite the larger model, v2 still runs on Mac M3 thanks to:
- ✅ **Gradient Checkpointing**: 50% memory reduction
- ✅ **Gradient Accumulation**: Effective batch size of 16
- ✅ **Batch size**: 1 (with accumulation)
- ✅ **Sequence length**: 128 tokens (manageable)
- ✅ **MPS device**: Apple Silicon GPU acceleration
- ✅ **No AMP on MPS**: Avoids compatibility issues
- ✅ **num_workers=0**: Single-process data loading

## 📁 Project Structure

```
v2/
├── config.py          # Improved configuration (15M params, 25 epochs)
├── model.py           # Transformer architecture (Multi-Head Attention)
├── dataset.py         # Dolly-15k dataset loader (configurable size)
├── train.py           # Training loop with wandb integration
├── inference.py       # CLI inference script
├── app.py             # Gradio web UI for interactive testing
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd v2
pip install -r requirements.txt
```

### 2. Configure wandb (Optional)
Create a `.env` file in the `transformer/` directory:
```bash
WANDB_BASE_URL=http://localhost:8080
WANDB_ENTITY=your-entity
WANDB_PROJECT=transformer-v2-improved
WANDB_API_KEY=your-api-key
```

### 3. Train the Model
```bash
python train.py
```

**Expected training time on Mac M3**:
- ~15,625 total steps (25 epochs × 625 steps/epoch)
- ~8-10 hours (depends on M3 variant)

### 4. Test with Gradio UI
```bash
python app.py --checkpoint checkpoints/best_model.pt
```

Open your browser at `http://localhost:7860`

### 5. CLI Inference
```bash
# Interactive chat mode
python inference.py --checkpoint checkpoints/best_model.pt --chat

# Single generation
python inference.py --checkpoint checkpoints/best_model.pt \
  --prompt "What is machine learning?" --max-tokens 100
```

## 📈 Training Metrics to Monitor

### Good Training Indicators
- **Train loss**: Should decrease smoothly from ~9.0 to <3.0
- **Validation loss**: Should decrease to <3.0 (good quality)
- **Perplexity**: Should decrease from ~8000 to <20
- **Sample outputs**: Should become coherent by epoch 10-15

### Warning Signs
- ⚠️ Val loss stops improving: May need lower learning rate
- ⚠️ Val loss increases: Overfitting, consider dropout
- ⚠️ OOM errors: Reduce batch_size or max_seq_length

## 🔧 Hyperparameter Tuning

If you want to experiment further, edit [config.py](config.py#L8-L50):

```python
# Model size (increase for better quality, but more memory)
d_model: int = 192        # Try: 256 for 25M params
n_heads: int = 6          # Try: 8 (must divide d_model)
n_layers: int = 4         # Try: 6 for deeper model

# Training (increase for better quality, but longer training)
epochs: int = 25          # Try: 30-50 if you have time
dataset_size: int = 10000 # Try: 15000 (full dataset)
learning_rate: float = 2e-4 # Try: 1e-4 for more stability
```

## 📊 Comparison: v1 vs v2

| Metric | v1 (TINY) | v2 (Improved) | Change |
|--------|-----------|---------------|--------|
| Parameters | 7M | 15M | +114% |
| d_model | 128 | 192 | +50% |
| n_heads | 4 | 6 | +50% |
| n_layers | 3 | 4 | +33% |
| d_ff | 512 | 768 | +50% |
| Epochs | 10 | 25 | +150% |
| Dataset samples | 5,000 | 10,000 | +100% |
| Total steps | 1,000 | 15,625 | +1462% |
| Learning rate | 3e-4 | 2e-4 | -33% |
| Expected val loss | ~6.62 | <3.0 | -55% |
| Output quality | Poor | Good | ✅ |

## 🎯 Use Cases

After training, this model can:
- Answer simple questions
- Follow basic instructions
- Generate short text completions
- Demonstrate transformer attention mechanisms

**Limitations**:
- Still a TINY model (15M params vs GPT-2's 117M)
- Limited knowledge (only 10k training samples)
- Best for educational purposes and experimentation

## 🐛 Troubleshooting

### Out of Memory (MPS)
```python
# In config.py, reduce these:
batch_size = 1
max_seq_length = 96  # Reduce from 128
dataset_size = 5000  # Reduce from 10000
```

### Training Too Slow
```python
# Speed up at cost of quality:
epochs = 15  # Reduce from 25
dataset_size = 5000  # Reduce from 10000
gradient_accumulation_steps = 8  # Reduce from 16
```

### Poor Output Quality
```python
# Increase capacity (requires more memory):
epochs = 30  # Increase from 25
dataset_size = 15000  # Use full dataset
learning_rate = 1e-4  # Lower for stability
```

## 📚 Learn More

- **Attention Is All You Need**: [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- **GPT-2 Paper**: [openai.com/research/better-language-models](https://openai.com/research/better-language-models)
- **databricks-dolly-15k**: [huggingface.co/datasets/databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

## 📝 License

Educational project for learning transformer architectures.

---

**v2 Release Notes**: Improved quality through larger model (15M params), more training (25 epochs), and better hyperparameters. Still optimized for Mac M3 compatibility.
