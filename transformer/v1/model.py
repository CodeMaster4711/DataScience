"""
Transformer LLM Implementation with Multi-Head Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from config import TransformerConfig


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, n_heads, seq_len, d_k)
            key: (batch_size, n_heads, seq_len, d_k)
            value: (batch_size, n_heads, seq_len, d_v)
            mask: (batch_size, 1, seq_len, seq_len) or (batch_size, n_heads, seq_len, seq_len)
        Returns:
            output: (batch_size, n_heads, seq_len, d_v)
            attention_weights: (batch_size, n_heads, seq_len, seq_len)
        """
        d_k = query.size(-1)

        # Compute attention scores: Q·K^T / √d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (for causal/autoregressive attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: Allows model to jointly attend to information
    from different representation subspaces.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads  # Dimension per head

        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)

        # Output projection
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)

        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=config.attention_dropout)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, 1, seq_len) for causal masking
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # Linear projections and split into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k)
        # -> (batch_size, n_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)

        # Concatenate heads: (batch_size, n_heads, seq_len, d_k)
        # -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Final linear projection
        output = self.W_o(attn_output)
        output = self.dropout(output)

        return output, attention_weights


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)

        # Create positional encoding matrix
        pe = torch.zeros(config.max_seq_length, config.d_model)
        position = torch.arange(0, config.max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_length, d_model)

        # Register as buffer (not a parameter, but part of the state)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    Uses GELU activation (like GPT-2/BERT)
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()  # GELU is smoother than ReLU

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer Block:
    - Multi-Head Self-Attention with residual connection & layer norm
    - Feed-Forward Network with residual connection & layer norm
    Uses Pre-LN (LayerNorm before sublayers) for better training stability
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.use_gradient_checkpointing = False

    def _forward_impl(self, x, mask):
        """Implementation of forward pass for checkpointing"""
        # Pre-LN: LayerNorm before attention
        normed_x = self.ln1(x)
        attn_output, attention_weights = self.attention(normed_x, normed_x, normed_x, mask)
        x = x + attn_output  # Residual connection

        # Pre-LN: LayerNorm before feed-forward
        normed_x = self.ln2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + ff_output  # Residual connection

        return x, attention_weights

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, 1, seq_len) for causal masking
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, n_heads, seq_len, seq_len)
        """
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            x.requires_grad_(True)
            return checkpoint(self._forward_impl, x, mask, use_reentrant=False)
        else:
            return self._forward_impl(x, mask)


class TransformerLLM(nn.Module):
    """
    Complete Transformer Language Model (Decoder-only, like GPT)
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = False

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(config)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer normalization
        self.ln_final = nn.LayerNorm(config.d_model)

        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights: share embeddings between input and output
        # (common practice in language models to reduce parameters)
        self.output_projection.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small random values (like GPT-2)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def create_causal_mask(self, seq_len, device):
        """
        Create causal mask for autoregressive generation:
        Prevents attending to future tokens
        Returns: (1, 1, seq_len, seq_len) mask
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.view(1, 1, seq_len, seq_len)

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - Token indices
            targets: (batch_size, seq_len) - Target tokens for loss computation (optional)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar (if targets provided)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Token embeddings
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Create causal mask
        mask = self.create_causal_mask(seq_len, device)

        # Pass through transformer blocks
        attention_weights_list = []
        for block in self.transformer_blocks:
            x, attention_weights = block(x, mask)
            attention_weights_list.append(attention_weights)

        # Final layer normalization
        x = self.ln_final(x)

        # Project to vocabulary
        logits = self.output_projection(x)  # (batch_size, seq_len, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (batch_size * seq_len, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100  # Ignore padding tokens
            )

        return logits, loss

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Generate text autoregressively
        Args:
            input_ids: (batch_size, seq_len) - Prompt tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (None = no filtering)
        Returns:
            generated_ids: (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop to max sequence length
                input_ids_crop = input_ids[:, -self.config.max_seq_length:]

                # Forward pass
                logits, _ = self.forward(input_ids_crop)

                # Get logits for last token
                logits = logits[:, -1, :] / temperature

                # Top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory (trades compute for memory)"""
        self.use_gradient_checkpointing = True
        for block in self.transformer_blocks:
            block.use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        self.use_gradient_checkpointing = False
        for block in self.transformer_blocks:
            block.use_gradient_checkpointing = False

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
