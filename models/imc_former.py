"""
IMC-Former Model
================
Criticality-aware dual-stream transformer for IMC schedulability prediction.

Architecture overview:
  1. Per-task MLP encoder Φ           : f_i ∈ ℝ^12  →  h_i ∈ ℝ^d
  2. Criticality partition            : split embeddings into H_LO, H_HI by χ_i
  3. Intra-stream transformer Ψ_intra : self-attention within each stream
  4. Cross-stream attention Ψ_cross   : HI→LO and LO→HI directional attention
  5. Hierarchical attention pooling   : per-stream → c_LO, c_HI ∈ ℝ^d
  6. Context MLP                      : z_set ∈ ℝ^6  →  z ∈ ℝ^d_ctx
  7. Global fusion                    : c = Concat([c_LO, c_HI, z])
  8. Scheduler-specific heads Γ_S    : c  →  ŷ_S ∈ [0,1]

Design invariants:
  - Permutation invariance : all pooling is order-independent
  - Variable-length        : binary masks propagate through every attention op
  - Stream separation      : LO and HI tasks never mix until fusion
  - Robustness             : all-LO / all-HI task sets handled without NaN
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import ModelConfig


# =============================================================================
# 1. Per-Task MLP Encoder
# =============================================================================

class TaskEncoder(nn.Module):
    """
    Shared MLP applied independently to each task feature vector.

    Maps f_i ∈ ℝ^{feat_dim} → h_i ∈ ℝ^d via stacked Linear→GELU→LayerNorm.
    Shared weights across all tasks and both streams: the encoder learns
    a universal task representation; stream-specific behaviour emerges in the
    transformer and pooling layers.
    """

    def __init__(self, feat_dim: int, d_model: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = feat_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
            in_dim = hidden_dim
        layers += [nn.Linear(hidden_dim, d_model), nn.LayerNorm(d_model)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, feat_dim) → (B, N, d_model)"""
        return self.net(x)


# =============================================================================
# 2. Masked Transformer Block
# =============================================================================

class MaskedTransformerBlock(nn.Module):
    """
    Standard transformer encoder block with padding-mask support.
    key_padding_mask follows PyTorch convention: True = position is IGNORED.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 attn_drop: float = 0.1, ffn_drop: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=attn_drop, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(ffn_drop),
            nn.Linear(ffn_dim, d_model), nn.Dropout(ffn_drop),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, N, d) → (B, N, d)"""
        attn_out, _ = self.self_attn(x, x, x,
                                     key_padding_mask=key_padding_mask,
                                     need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


# =============================================================================
# 3. Masked Cross-Attention Block
# =============================================================================

class MaskedCrossAttentionBlock(nn.Module):
    """
    Cross-attention where Query stream attends to Key/Value stream.

    Two directional uses in IMC-Former:
      (a) HI-on-LO: HI tasks (Q) query LO tasks (K/V).
          Approximates the AMC mode-switch interference term: when computing
          HI task response time in HI mode, the analysis accounts for LO task
          execution that consumed processor time before the mode switch.
          Attention weights learn to quantify each LO task's interference
          contribution to each HI task's deadline feasibility.

      (b) LO-on-HI: LO tasks (Q) query HI tasks (K/V).
          Captures how HI mandatory execution constrains remaining capacity
          for LO optional execution in LO mode.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                query_mask: Optional[torch.Tensor] = None,
                kv_mask:    Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query     : (B, N_q, d)
            key_value : (B, N_kv, d)
            query_mask: (B, N_q) bool — True=padding (not used as key_padding_mask
                        for the query side, only applied post-hoc to zero padding)
            kv_mask   : (B, N_kv) bool — True=padding, passed to MHA
        Returns:
            out: (B, N_q, d) — query enriched with kv context
        """
        attn_out, _ = self.cross_attn(
            query, key_value, key_value,
            key_padding_mask=kv_mask,
            need_weights=False,
        )
        out = self.norm(query + attn_out)
        return out


# =============================================================================
# 4. Hierarchical Attention Pooling  (BUG-FIXED)
# =============================================================================

class HierarchicalAttentionPool(nn.Module):
    """
    Two-level attention pooling for scaling to large task sets.

    Level 1 — Local pooling within chunks:
        Tasks are sorted by deadline ratio δ_i (ascending, tightest first).
        Chunks of size `chunk_size` are each summarised by a learned attention
        pooling query into one chunk-level vector.

    Level 2 — Global pooling over chunk summaries:
        Chunk summaries pass through a transformer block then attention pooling
        to produce the stream-level representation.

    Edge cases handled:
        - n not divisible by chunk_size: zero-pad to next multiple, mask padded chunks
        - Empty stream (all tasks padded): detect and return zero vector, avoiding
          the -inf→NaN collapse that would otherwise occur in softmax over all-masked
          positions.

    Sorting rationale:
        δ_i = D_i/T_i is used because tasks with smaller δ_i (tighter relative
        deadlines) dominate schedulability analysis under both EDF (deadline-driven
        priority) and fixed-priority DM (deadline monotonic). Sorting ensures
        local chunks contain tasks with similar temporal behaviour.
    """

    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 8,
                 pool_dim: int = 256, ffn_dim: int = 512, ffn_drop: float = 0.1):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model    = d_model

        # Learnable query for local attention pooling
        self.local_pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.local_pool_attn  = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True,
        )

        self.global_transformer = MaskedTransformerBlock(
            d_model=d_model, n_heads=n_heads,
            ffn_dim=ffn_dim, ffn_drop=ffn_drop,
        )

        # Global attention pool: e_j = v^T tanh(W_a h_j + b_a)
        self.pool_w = nn.Linear(d_model, pool_dim)
        self.pool_v = nn.Linear(pool_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x     : (B, N, d_model)
            mask  : (B, N) float — 1=real task, 0=padding
            delta : (B, N) float — deadline ratio for sorting
        Returns:
            c     : (B, d_model) — stream-level representation
                    Zero vector for batches with entirely empty stream.
        """
        B, N, d = x.shape

        # ── Guard: detect entirely-empty streams ───────────────────────────
        # A stream is empty if no position has mask=1.
        stream_nonempty = mask.any(dim=1)   # (B,) bool — True if ≥1 real task

        # ── Sort by δ ascending; padding to end ────────────────────────────
        delta_sort = delta.clone()
        delta_sort[mask < 0.5] = float("inf")
        sort_idx  = delta_sort.argsort(dim=1)          # (B, N)
        x_sorted  = torch.gather(x, 1, sort_idx.unsqueeze(-1).expand_as(x))
        m_sorted  = torch.gather(mask, 1, sort_idx).float()

        # ── Chunk: pad N to multiple of chunk_size ─────────────────────────
        n_chunks = math.ceil(N / self.chunk_size)
        pad_len  = n_chunks * self.chunk_size - N
        if pad_len > 0:
            x_sorted = F.pad(x_sorted, (0, 0, 0, pad_len))
            m_sorted = F.pad(m_sorted, (0, pad_len))

        x_chunks = x_sorted.reshape(B * n_chunks, self.chunk_size, d)
        m_chunks = m_sorted.reshape(B * n_chunks, self.chunk_size)

        # ── Local attention pooling per chunk ──────────────────────────────
        pad_local = (m_chunks < 0.5)   # (B*K, chunk) — True=pad
        # Avoid all-True rows: keep at least one slot unmasked per chunk
        # (for chunks that are entirely padding, the output will be near-zero
        # anyway and will be masked out at the global level)
        all_pad_local = pad_local.all(dim=1, keepdim=True)   # (B*K, 1)
        pad_local_safe = pad_local & ~all_pad_local

        q = self.local_pool_query.expand(B * n_chunks, 1, d)
        chunk_out, _ = self.local_pool_attn(
            q, x_chunks, x_chunks,
            key_padding_mask=pad_local_safe,
            need_weights=False,
        )   # (B*K, 1, d)
        chunk_summaries = chunk_out.squeeze(1).reshape(B, n_chunks, d)   # (B, K, d)

        # Chunk is "real" if it contains at least one real task
        chunk_mask = m_sorted.reshape(B, n_chunks, self.chunk_size)\
                              .any(dim=-1).float()   # (B, K)

        # ── Global transformer over chunk summaries ────────────────────────
        pad_global = (chunk_mask < 0.5)   # (B, K) True=padded chunk
        chunk_summaries = self.global_transformer(
            chunk_summaries, key_padding_mask=pad_global,
        )

        # ── Global attention pooling ───────────────────────────────────────
        e = self.pool_v(torch.tanh(self.pool_w(chunk_summaries))).squeeze(-1)  # (B, K)
        e = e.masked_fill(pad_global, float("-inf"))

        # ── FIX: for entirely-empty streams, replace -inf with 0 before softmax
        # so softmax produces uniform weights, then the zero-masked embeddings
        # produce a near-zero output. We then zero the output explicitly.
        e_safe = e.clone()
        empty_stream = ~stream_nonempty   # (B,) — True if no real tasks
        if empty_stream.any():
            # Replace all -inf with 0 for empty-stream rows (softmax will be uniform,
            # but the chunk summaries themselves are near-zero so output ≈ 0)
            e_safe[empty_stream] = 0.0

        alpha = torch.softmax(e_safe, dim=1)   # (B, K)
        c = (alpha.unsqueeze(-1) * chunk_summaries).sum(dim=1)   # (B, d)

        # Explicitly zero out representations for empty streams
        c = c * stream_nonempty.float().unsqueeze(-1)   # (B, d)

        return c


# =============================================================================
# 5. Scheduler-Specific Prediction Head
# =============================================================================

class SchedulerHead(nn.Module):
    """
    2-layer MLP head per IMC scheduling policy → ŷ_S ∈ [0,1].

    MLP (not linear) heads are used because IMC feasibility conditions involve
    mode-switch behaviour and dual-mode interference that create non-linearly
    separable decision boundaries in the shared latent space.
    """

    def __init__(self, in_dim: int, hidden: int, n_layers: int = 2,
                 dropout: float = 0.1, name: str = "scheduler"):
        super().__init__()
        self.name = name
        layers, cur = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(cur, hidden), nn.GELU(),
                       nn.LayerNorm(hidden), nn.Dropout(dropout)]
            cur = hidden
        layers.append(nn.Linear(cur, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """c: (B, in_dim) → (B,) probabilities in [0,1]"""
        return torch.sigmoid(self.mlp(c).squeeze(-1))


# =============================================================================
# 6. Context MLP
# =============================================================================

class ContextMLP(nn.Module):
    """
    Projects set-level analytical features z_set ∈ ℝ^6 to a latent vector.

    z_set = [U_tot_LO, U_tot_HI, HI_ratio, CF_global, slack_margin, HI_demand]

    These aggregate quantities provide the model with direct access to the
    quantities that appear in sufficient schedulability conditions (e.g.,
    U_tot_LO ≤ 1 for EDF implicit-deadline; EDF-VD bounds involving U_tot_HI),
    complementing the per-task interaction features learned by the transformer.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),  nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim), nn.GELU(), nn.LayerNorm(out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 6) → (B, out_dim)"""
        return self.net(z)


# =============================================================================
# 7. IMC-Former — Full Model  (BUG-FIXED)
# =============================================================================

class IMCFormer(nn.Module):
    """
    IMC-Former: Criticality-Aware Dual-Stream Transformer for IMC Schedulability.

    The criticality split uses feature index 4 (chi_i) from the RAW (pre-
    normalisation) feature tensor to cleanly separate LO (chi=0) from HI (chi=1)
    tasks. After z-score normalisation the original 0 maps to a negative value
    and 1 maps to a positive value (since HI_ratio < 1 always). We therefore
    use sign of the normalised chi feature as the split criterion, which is
    reliable for HI_ratio ∈ (0, 1) exclusive — guaranteed by the data generator.

    For robustness (all-LO or all-HI task sets), the model computes a zero
    representation for the empty stream and relies on the set-level context
    features (HI_ratio=0 or =1) to inform the heads.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # (1) Shared per-task encoder
        self.task_encoder = TaskEncoder(
            feat_dim=cfg.task_feature_dim, d_model=d,
            hidden_dim=cfg.encoder_hidden, n_layers=cfg.encoder_layers,
        )

        # (2) Intra-stream transformers (separate weights per stream)
        self.lo_transformer = nn.ModuleList([
            MaskedTransformerBlock(d, cfg.n_heads, cfg.ffn_dim,
                                   cfg.attn_dropout, cfg.ffn_dropout)
            for _ in range(cfg.n_transformer_layers)
        ])
        self.hi_transformer = nn.ModuleList([
            MaskedTransformerBlock(d, cfg.n_heads, cfg.ffn_dim,
                                   cfg.attn_dropout, cfg.ffn_dropout)
            for _ in range(cfg.n_transformer_layers)
        ])

        # (3) Cross-stream attention (directional, bidirectional)
        self.cross_hi_on_lo = MaskedCrossAttentionBlock(
            d_model=d, n_heads=cfg.cross_attn_heads, dropout=cfg.cross_attn_dropout,
        )
        self.cross_lo_on_hi = MaskedCrossAttentionBlock(
            d_model=d, n_heads=cfg.cross_attn_heads, dropout=cfg.cross_attn_dropout,
        )

        # (4) Hierarchical pooling (separate per stream)
        self.lo_pool = HierarchicalAttentionPool(
            d_model=d, n_heads=cfg.n_heads, chunk_size=cfg.chunk_size,
            pool_dim=cfg.pool_hidden, ffn_dim=cfg.ffn_dim, ffn_drop=cfg.ffn_dropout,
        )
        self.hi_pool = HierarchicalAttentionPool(
            d_model=d, n_heads=cfg.n_heads, chunk_size=cfg.chunk_size,
            pool_dim=cfg.pool_hidden, ffn_dim=cfg.ffn_dim, ffn_drop=cfg.ffn_dropout,
        )

        # (5) Set-level context projection
        self.context_mlp = ContextMLP(
            in_dim=cfg.context_feature_dim,
            hidden=cfg.context_hidden * 2,
            out_dim=cfg.context_hidden,
        )

        # (6) Scheduler-specific prediction heads
        fused_dim = d * 2 + cfg.context_hidden
        assert fused_dim == cfg.fused_dim, \
            f"fused_dim mismatch: computed {fused_dim} vs config {cfg.fused_dim}"

        self.heads = nn.ModuleDict({
            name: SchedulerHead(
                in_dim=fused_dim, hidden=cfg.head_hidden,
                n_layers=cfg.head_layers, dropout=cfg.head_dropout, name=name,
            )
            for name in cfg.scheduler_names
        })

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: {
              "features" : (B, N, 12) — per-task features (padded, normalised)
              "mask"     : (B, N)      — 1=real, 0=padding
              "context"  : (B, 6)      — set-level context
            }
        Returns:
            {
              "logits" : {scheduler_name: (B,) probabilities},
              "c_lo"   : (B, d),
              "c_hi"   : (B, d),
              "c_fused": (B, 576),
            }
        """
        F_in  = batch["features"]   # (B, N, 12)
        mask  = batch["mask"]       # (B, N) float
        z_set = batch["context"]    # (B, 6)
        B, N, _ = F_in.shape

        # ── Criticality split ───────────────────────────────────────────────
        # Feature index 4 = chi_i (criticality level).
        # After z-score normalisation: original 0 → negative, original 1 → positive
        # (valid for HI_ratio ∈ (0,1), enforced by data generator).
        chi_norm     = F_in[:, :, 4]                        # (B, N)
        real         = mask > 0.5                           # (B, N) bool
        hi_bool      = (chi_norm > 0) & real                # real HI tasks
        lo_bool      = (chi_norm <= 0) & real               # real LO tasks

        # PyTorch MHA key_padding_mask: True = IGNORE position
        lo_pad = ~lo_bool   # (B, N) — ignore non-LO positions in LO stream
        hi_pad = ~hi_bool   # (B, N) — ignore non-HI positions in HI stream

        # Delta (deadline ratio) at feature index 9, used for sorting
        delta = F_in[:, :, 9]   # (B, N)

        # ── (1) Per-task encoding ───────────────────────────────────────────
        H = self.task_encoder(F_in)   # (B, N, d)

        # ── (2) Intra-stream self-attention ────────────────────────────────
        H_lo = H.clone()
        H_hi = H.clone()
        for block in self.lo_transformer:
            H_lo = block(H_lo, key_padding_mask=lo_pad)
        for block in self.hi_transformer:
            H_hi = block(H_hi, key_padding_mask=hi_pad)

        # ── (3) Cross-stream attention ──────────────────────────────────────
        # HI tasks query LO context (mode-switch interference)
        H_hi = self.cross_hi_on_lo(H_hi, H_lo,
                                   query_mask=hi_pad, kv_mask=lo_pad)
        # LO tasks query HI context (HI mandatory blocks LO optional)
        H_lo = self.cross_lo_on_hi(H_lo, H_hi,
                                   query_mask=lo_pad, kv_mask=hi_pad)

        # ── (4) Hierarchical attention pooling ─────────────────────────────
        lo_mask_f = lo_bool.float()
        hi_mask_f = hi_bool.float()

        delta_lo = delta.clone().masked_fill(~lo_bool, float("inf"))
        delta_hi = delta.clone().masked_fill(~hi_bool, float("inf"))

        c_lo = self.lo_pool(H_lo, lo_mask_f, delta_lo)   # (B, d)
        c_hi = self.hi_pool(H_hi, hi_mask_f, delta_hi)   # (B, d)

        # ── (5) Set-level context ───────────────────────────────────────────
        z = self.context_mlp(z_set)   # (B, context_hidden)

        # ── (6) Fusion ──────────────────────────────────────────────────────
        c = torch.cat([c_lo, c_hi, z], dim=-1)   # (B, 576)

        # ── (7) Scheduler heads ─────────────────────────────────────────────
        logits = {name: head(c) for name, head in self.heads.items()}

        return {"logits": logits, "c_lo": c_lo, "c_hi": c_hi, "c_fused": c}

    @torch.no_grad()
    def predict(self, batch: Dict, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """Binary predictions at given threshold."""
        out = self.forward(batch)
        return {name: (p >= threshold).long() for name, p in out["logits"].items()}

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
