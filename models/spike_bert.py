"""
SpikeBERT: Spiking Transformer for NLP tasks.

A BERT-style transformer where all activations pass through LIF neurons,
enabling spike-driven inference. Uses LayerNorm (not BatchNorm) and
operates on token sequences rather than images.

Input: (B, L) token IDs
Output: (B, num_classes) classification logits

Reference: https://github.com/Lvchangze/SpikeBERT
"""

import math

import torch
import torch.nn as nn

from models.layers import SeqToANNContainer
from models.neurons import MultiStepLIFNeuron


class SpikeBertSSA(nn.Module):
    """Spiking Self-Attention for SpikeBERT.

    Pattern: Linear → LN → LIF for Q, K, V projections,
    then scaled dot-product attention, then attn_lif.

    Attribute naming follows SparseSNN conventions for TDL detection:
    q_lif, k_lif, v_lif, attn_lif, num_heads.
    """

    def __init__(self, dim, num_heads, T, tau=2.0, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or (self.head_dim ** -0.5)
        self.T = T

        # Q/K/V projections: Linear → LayerNorm → LIF
        self.q_linear = SeqToANNContainer(nn.Linear(dim, dim))
        self.q_ln = SeqToANNContainer(nn.LayerNorm(dim))
        self.q_lif = MultiStepLIFNeuron(tau=tau, v_threshold=0.5, detach_reset=True)

        self.k_linear = SeqToANNContainer(nn.Linear(dim, dim))
        self.k_ln = SeqToANNContainer(nn.LayerNorm(dim))
        self.k_lif = MultiStepLIFNeuron(tau=tau, v_threshold=0.5, detach_reset=True)

        self.v_linear = SeqToANNContainer(nn.Linear(dim, dim))
        self.v_ln = SeqToANNContainer(nn.LayerNorm(dim))
        self.v_lif = MultiStepLIFNeuron(tau=tau, v_threshold=0.5, detach_reset=True)

        # Post-attention LIF
        self.attn_lif = MultiStepLIFNeuron(tau=tau, v_threshold=0.5, detach_reset=True)

        # Output projection
        self.proj = SeqToANNContainer(nn.Linear(dim, dim))
        self.proj_ln = SeqToANNContainer(nn.LayerNorm(dim))
        self.proj_lif = MultiStepLIFNeuron(tau=tau, v_threshold=0.5, detach_reset=True)

    def forward(self, x):
        """
        Args:
            x: (T, B, L, D) input sequence

        Returns:
            (T, B, L, D) attention output
        """
        T, B, L, D = x.shape

        # Q/K/V projections
        q = self.q_lif(self.q_ln(self.q_linear(x)))   # (T, B, L, D)
        k = self.k_lif(self.k_ln(self.k_linear(x)))
        v = self.v_lif(self.v_ln(self.v_linear(x)))

        # Reshape to multi-head: (T, B, L, D) → (T*B, heads, L, head_dim)
        q = q.flatten(0, 1).view(-1, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.flatten(0, 1).view(-1, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.flatten(0, 1).view(-1, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (T*B, heads, L, L)
        x = attn @ v                                     # (T*B, heads, L, hd)

        # Merge heads: (T*B, heads, L, hd) → (T, B, L, D)
        x = x.transpose(1, 2).reshape(-1, L, D)
        x = x.view(T, B, L, D)

        # attn_lif + projection
        x = self.attn_lif(x)
        x = self.proj_lif(self.proj_ln(self.proj(x)))
        return x


class SpikeBertMLP(nn.Module):
    """Spiking MLP block for SpikeBERT.

    Pattern: Linear → LN → LIF → Linear → LN → LIF
    """

    def __init__(self, dim, hidden_dim=None, T=4, tau=2.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.fc1 = SeqToANNContainer(nn.Linear(dim, hidden_dim))
        self.ln1 = SeqToANNContainer(nn.LayerNorm(hidden_dim))
        self.lif1 = MultiStepLIFNeuron(tau=tau, v_threshold=0.5, detach_reset=True)

        self.fc2 = SeqToANNContainer(nn.Linear(hidden_dim, dim))
        self.ln2 = SeqToANNContainer(nn.LayerNorm(dim))
        self.lif2 = MultiStepLIFNeuron(tau=tau, v_threshold=0.5, detach_reset=True)

    def forward(self, x):
        """x: (T, B, L, D) → (T, B, L, D)"""
        x = self.lif1(self.ln1(self.fc1(x)))
        x = self.lif2(self.ln2(self.fc2(x)))
        return x


class SpikeBertBlock(nn.Module):
    """Single SpikeBERT transformer block: SSA + MLP with residual."""

    def __init__(self, dim, num_heads, T, mlp_ratio=4, tau=2.0):
        super().__init__()
        self.attn = SpikeBertSSA(dim, num_heads, T, tau=tau)
        self.mlp = SpikeBertMLP(dim, int(dim * mlp_ratio), T, tau=tau)

    def forward(self, x):
        """x: (T, B, L, D) → (T, B, L, D)"""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SpikeBERT(nn.Module):
    """SpikeBERT: Spiking BERT for text classification.

    Input: (B, L) token IDs (long tensor)
    Output: (B, num_classes) classification logits

    Args:
        vocab_size: vocabulary size (default 30522 for BERT)
        num_classes: number of classification classes
        hidden_dim: transformer hidden dimension
        num_heads: number of attention heads
        num_layers: number of transformer blocks
        max_seq_len: maximum sequence length
        T: number of temporal steps
        tau: LIF neuron time constant
        mlp_ratio: MLP hidden dim ratio
    """

    def __init__(self, vocab_size=30522, num_classes=2, hidden_dim=768,
                 num_heads=12, num_layers=12, max_seq_len=512, T=4,
                 tau=2.0, mlp_ratio=4, in_channels=None, img_size=None,
                 **kwargs):
        super().__init__()
        self.T = T
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Token + position embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SpikeBertBlock(hidden_dim, num_heads, T, mlp_ratio, tau)
            for _ in range(num_layers)
        ])

        # Classification head
        self.head_ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, L) token IDs (long or int tensor)

        Returns:
            (B, num_classes) classification logits
        """
        B, L = x.shape
        # Ensure valid token indices (handles float dummy input from ONNX tracing)
        x = x.long().clamp(0, self.embedding.num_embeddings - 1)

        # Embedding: (B, L) → (B, L, D)
        x = self.embedding(x) + self.pos_embedding[:, :L, :]

        # Repeat across T timesteps: (B, L, D) → (T, B, L, D)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Temporal mean: (T, B, L, D) → (B, L, D)
        x = x.mean(dim=0)

        # Pool over sequence: (B, L, D) → (B, D)
        x = x.mean(dim=1)

        # Classifier
        x = self.head(self.head_ln(x))
        return x


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def spike_bert_base(num_classes=2, T=4, **kwargs):
    """SpikeBERT-Base (768d, 12 heads, 12 layers)."""
    return SpikeBERT(num_classes=num_classes, hidden_dim=768,
                     num_heads=12, num_layers=12, T=T, **kwargs)


def spike_bert_small(num_classes=2, T=4, **kwargs):
    """SpikeBERT-Small (256d, 4 heads, 4 layers)."""
    return SpikeBERT(num_classes=num_classes, hidden_dim=256,
                     num_heads=4, num_layers=4, T=T, **kwargs)


def build_spike_bert(config):
    """Build SpikeBERT from config dict."""
    return SpikeBERT(
        vocab_size=config.get('vocab_size', 30522),
        num_classes=config.get('num_classes', 2),
        hidden_dim=config.get('hidden_dim', 768),
        num_heads=config.get('num_heads', 12),
        num_layers=config.get('num_layers', 12),
        max_seq_len=config.get('max_seq_len', 512),
        T=config.get('T', 4),
        tau=config.get('tau', 2.0),
        mlp_ratio=config.get('mlp_ratio', 4),
    )
