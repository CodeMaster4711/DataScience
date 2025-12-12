# Transformer LLM - From Scratch Implementation

Ein vollständiges Transformer Language Model (LLM) von Grund auf implementiert in PyTorch mit Multi-Head Attention, trainiert auf dem databricks-dolly-15k Dataset.

## Architektur

### Multi-Head Attention Layer
- **Scaled Dot-Product Attention**: `Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V`
- **8 parallele Attention Heads**: Jeder Head lernt unterschiedliche Aspekte der Sequenz
- **Causal Masking**: Für autoregressive Text-Generierung (verhindert "cheating")

### Komponenten
1. **Token Embeddings**: Konvertiert Token IDs zu Vektoren
2. **Positional Encoding**: Sinusoidal encoding für Positions-Information
3. **Transformer Blocks** (6 Schichten):
   - Multi-Head Self-Attention
   - Feed-Forward Network (2-layer MLP mit GELU)
   - Layer Normalization (Pre-LN)
   - Residual Connections
4. **Output Layer**: Linear Projection zu Vocabulary

### Hyperparameter
```python
vocab_size: 50257        # GPT-2 Tokenizer
d_model: 512            # Embedding dimension
n_heads: 8              # Attention heads
n_layers: 6             # Transformer blocks
d_ff: 2048             # Feed-forward dimension
max_seq_length: 512    # Maximum sequence length
dropout: 0.1
batch_size: 8
learning_rate: 3e-4
epochs: 10
```

## Dataset: databricks-dolly-15k

Instruction-following dataset mit 15k Samples:
- **Format**: `{"instruction": "...", "context": "...", "response": "..."}`
- **Preprocessing**: `Instruction: {instruction}\nContext: {context}\nResponse: {response}`
- **Tokenizer**: GPT-2 Tokenizer (50257 vocab size)

## Setup

### 1. Dependencies installieren
```bash
cd transformer/v1
pip install -r requirements.txt
```

### 2. .env Datei erstellen
```bash
# Copy .env.example to .env
cp ../../.env.example .env

# Edit .env und fülle deine wandb API Key ein
# WANDB_API_KEY=your_api_key_here
# Uncomment: WANDB_PROJECT=transformer-llm
```

### 3. Training starten
```bash
python train.py
```

Das Training wird:
- Dataset automatisch von HuggingFace laden
- Model initialisieren (~38M Parameter)
- Training mit wandb logging starten
- Checkpoints speichern in `checkpoints/`
- Sample Text während Training generieren

## Verwendung

### Training
```bash
# Standard Training
python train.py

# Training auf bestimmtem Device
python train.py  # auto-detect: cuda > mps > cpu
```

### Inference / Text Generation

#### Interaktiver Chat Mode
```bash
python inference.py --checkpoint checkpoints/best_model.pt --chat
```

#### Single Prompt
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Instruction: What is Python?\nResponse:" \
    --max-tokens 100 \
    --temperature 0.8
```

#### Demo Mode (Example Prompts)
```bash
python inference.py --checkpoint checkpoints/best_model.pt
```

### Parameter für Text-Generierung
- `--temperature`: 0.1 (deterministisch) bis 2.0 (kreativ)
- `--top-k`: Top-k sampling (z.B. 40)
- `--max-tokens`: Maximale Anzahl generierter Tokens

## Monitoring mit Weights & Biases

wandb tracked:
- **Training Metrics**: Loss, Perplexity, Learning Rate
- **Validation Metrics**: Loss, Perplexity
- **Sample Generations**: Text Samples während Training
- **Model Architecture**: Layer Visualisierung
- **Gradients**: Gradient Flow Monitoring

Dashboard: `https://wandb.ai/{WANDB_ENTITY}/transformer-llm`

## Projektstruktur

```
transformer/v1/
├── config.py           # Hyperparameter & Konfiguration
├── model.py            # Transformer Architektur
│   ├── ScaledDotProductAttention
│   ├── MultiHeadAttention
│   ├── PositionalEncoding
│   ├── FeedForward
│   ├── TransformerBlock
│   └── TransformerLLM
├── dataset.py          # Dolly-15k Dataset Loader
├── train.py            # Training Loop mit wandb
├── inference.py        # Text Generation
├── requirements.txt    # Dependencies
├── README.md          # Diese Datei
└── checkpoints/       # Model Checkpoints (wird erstellt)
```

## Model Details

### Parameter Count
- Embedding Layer: ~25.7M parameters
- Transformer Blocks (6x): ~12.6M parameters
- **Total**: ~38M parameters (trainable)

### Attention Mechanism
Jeder Attention Head hat Dimension `d_k = d_model / n_heads = 512 / 8 = 64`

**Self-Attention Berechnung**:
1. Linear Projections: Q, K, V = W_q·x, W_k·x, W_v·x
2. Split in 8 Heads: (batch, seq_len, d_model) → (batch, 8, seq_len, 64)
3. Scaled Dot-Product: scores = Q·K^T / √64
4. Apply Causal Mask: mask future tokens
5. Softmax: attention_weights = softmax(scores)
6. Apply to Values: output = attention_weights · V
7. Concatenate Heads & Project: output = W_o · concat(heads)

## Performance Tipps

### Für schnelleres Training:
- **GPU**: Nutze CUDA wenn verfügbar
- **MPS**: Apple Silicon (M1/M2) nutzt MPS backend
- **Batch Size**: Erhöhe wenn genug VRAM (8 → 16 → 32)
- **Mixed Precision**: Aktiviere für schnelleres Training (TODO)

### Für bessere Results:
- **Mehr Epochs**: 10 → 20+ epochs
- **Größeres Model**: n_layers: 6 → 12, d_model: 512 → 768
- **Learning Rate Schedule**: Warmup + Cosine Decay (bereits implementiert)
- **Data Augmentation**: Paraphrasen, Backtranslation

## Troubleshooting

### CUDA Out of Memory
- Reduziere `batch_size` in [config.py](config.py)
- Reduziere `max_seq_length`
- Nutze Gradient Accumulation (TODO)

### Slow Training
- Check Device: sollte CUDA oder MPS sein
- Reduziere `eval_interval` und `save_interval`
- Nutze `num_workers` in DataLoader

### Poor Generation Quality
- Model noch nicht genug trainiert (train longer)
- Adjust `temperature` (0.7-0.9 ist gut)
- Adjust `top_k` (30-50)

## Weiterentwicklung

Mögliche Erweiterungen:
- [ ] Flash Attention für schnelleres Training
- [ ] Mixed Precision Training (FP16/BF16)
- [ ] Gradient Accumulation für größere Batches
- [ ] Rotary Position Embeddings (RoPE)
- [ ] Multi-Query Attention (MQA)
- [ ] KV-Cache für schnellere Inference
- [ ] Model Quantization (INT8)
- [ ] Fine-tuning auf eigenen Daten

## Referenzen

- Vaswani et al. (2017): "Attention Is All You Need"
- Radford et al. (2019): "Language Models are Unsupervised Multitask Learners" (GPT-2)
- databricks-dolly-15k: https://huggingface.co/datasets/databricks/databricks-dolly-15k
