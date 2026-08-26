"""
IMC-Former Safety-Aware Loss
=============================
The loss function has two terms:

  L = Σ_S w_S · BCE(y_S, ŷ_S)
    + λ · (1/B) Σ_b Σ_S (1 - y_{b,S}) · ŷ_{b,S}
    + μ · (1/B) Σ_b Σ_S (1 - y_{b,S}) · ŷ_{b,S} · HI_critical_b

Term 1 — Weighted Binary Cross-Entropy (BCE):
  Standard classification loss, scheduler-weighted.
  Handles class imbalance via per-scheduler BCE weights derived from data.
  Uses F.binary_cross_entropy_with_logits (numerically stable logit-space BCE)
  rather than F.binary_cross_entropy on sigmoid probabilities.
  This eliminates the sigmoid-saturation → log(0) → NaN gradient explosion
  that caused the CUDA assertion crash when logits exceeded ~88 in absolute value.

Term 2 — False-Positive Penalty:
  For any example where y_{b,S} = 0 (unschedulable) and ŷ_{b,S} is high,
  the model incurs an additional loss proportional to ŷ_{b,S}.
  When y=0 (unschedulable): penalty = ŷ   (penalise confidence in wrong prediction)
  When y=1 (schedulable):   penalty = 0   (no penalty; this term only fires on FPs)
  Controlled by λ (fp_penalty_lambda).

Term 3 — HI-Critical False-Positive Penalty:
  An ADDITIONAL penalty specifically for task sets where any HI task has
  U_HI_i above a threshold (default 0.7). These represent task sets near the
  HI-mode feasibility boundary where a false positive (predicting schedulable)
  is most dangerous. Controlled by μ (hi_fp_penalty_mu).

Note: No consistency regularization across schedulers is included (pending
mathematical verification of the partial ordering between schedulers).
Note: No utilization-margin focal loss is included (pending justification).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from configs.config import LossConfig


# ── Numerical safety constants ─────────────────────────────────────────────────
# Clamp logits before sigmoid-based ops to prevent ±inf and the resulting NaN
# in gradients. These bounds correspond to sigmoid(88) ≈ 1-2e-39 and
# sigmoid(-88) ≈ 2e-39, i.e. saturated-but-finite probabilities.
_LOGIT_CLAMP = 80.0

# Clamp probabilities for the FP-penalty terms (which multiply ŷ directly rather
# than using log(ŷ)), to prevent any residual NaN from upstream operations.
_PROB_EPS = 1e-7


class IMCLoss(nn.Module):
    """
    Safety-aware loss for IMC schedulability prediction.

    KEY CHANGE vs. previous version:
        F.binary_cross_entropy(sigmoid(logit), y) is replaced by
        F.binary_cross_entropy_with_logits(logit, y), which computes
        BCE in numerically stable logit space:
            BCE = max(logit, 0) - logit*y + log(1 + exp(-|logit|))
        This avoids the log(0) / 1/0 gradient catastrophe that occurs when
        sigmoid(logit) saturates to exactly 0.0 or 1.0 in float32 (logit > 88).

        The SchedulerHead still outputs sigmoid(logit) for inference and for
        the FP-penalty terms; we simply route raw logits to the BCE term separately.

    Args:
        cfg             : LossConfig with λ, μ, thresholds, and scheduler weights
        scheduler_names : list of scheduler name strings (must match model head names)
        pos_weights     : optional per-scheduler positive class weights (n_neg/n_pos).
                          Shape (4,) or None.
    """

    def __init__(
        self,
        cfg: LossConfig,
        scheduler_names: List[str],
        pos_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.cfg             = cfg
        self.scheduler_names = scheduler_names
        n_sched              = len(scheduler_names)

        self.register_buffer(
            "sched_weights",
            torch.tensor(cfg.scheduler_weights, dtype=torch.float32),
        )

        if pos_weights is not None:
            self.register_buffer("pos_weights", pos_weights)
        else:
            self.register_buffer("pos_weights", torch.ones(n_sched))

        # Clamp pos_weights to [0.1, 10.0] to prevent extreme gradient magnitudes
        # when the dataset is severely imbalanced (e.g. TT_Merge at 0.058 before fix).
        # The v2 balanced dataset makes this moot, but the guard remains as a safety net.
        self.register_buffer(
            "pos_weights_clamped",
            self.pos_weights.clamp(min=0.1, max=10.0),
        )

        self.lam       = cfg.fp_penalty_lambda
        self.mu        = cfg.hi_fp_penalty_mu
        self.hi_thresh = cfg.hi_critical_threshold

    def forward(
        self,
        logits:   Dict[str, torch.Tensor],   # raw logits AND probabilities
        labels:   torch.Tensor,              # (B, 4)
        features: torch.Tensor,              # (B, N, 12)
        mask:     torch.Tensor,              # (B, N)
        raw_logits: Dict[str, torch.Tensor] = None,  # (B,) pre-sigmoid logits
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits      : dict {scheduler_name: (B,) probabilities in [0,1]}
                          (sigmoid output from SchedulerHead)
            labels      : (B, 4) float — ground-truth schedulability {0, 1}
            features    : (B, N, 12) — per-task features (normalised)
            mask        : (B, N) float — 1=real task, 0=padding
            raw_logits  : dict {name: (B,)} pre-sigmoid values for stable BCE.
                          If None, BCE falls back to probability-space (less stable).

        Returns:
            dict: total, bce, fp_penalty, hi_fp_penalty, bce_per_sched
        """
        bce_terms    = []
        fp_terms     = []
        hi_fp_terms  = []

        # ── Identify HI-critical task sets ──────────────────────────────────────
        chi_norm   = features[:, :, 4]               # (B, N) normalised criticality
        u_hi_feat  = features[:, :, 6]               # (B, N) normalised U_HI
        real       = mask > 0.5
        hi_tasks   = (chi_norm > 0) & real
        u_hi_max   = (u_hi_feat * hi_tasks.float()).max(dim=1).values
        hi_critical = (u_hi_max > self.hi_thresh).float()

        # ── Per-scheduler terms ──────────────────────────────────────────────────
        for s_idx, name in enumerate(self.scheduler_names):
            ŷ_prob = logits[name]         # (B,) in [0,1]
            y      = labels[:, s_idx]    # (B,) in {0.0, 1.0}
            w      = self.sched_weights[s_idx]
            pw     = self.pos_weights_clamped[s_idx]

            # ── Numerically stable BCE ──────────────────────────────────────────
            # Prefer logit-space BCE (avoids log(0) in backward).
            if raw_logits is not None and name in raw_logits:
                logit = raw_logits[name].clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
                bce   = F.binary_cross_entropy_with_logits(
                    logit, y,
                    pos_weight=pw.unsqueeze(0).expand_as(logit),
                    reduction="mean",
                )
            else:
                # Fallback: clamp probabilities away from 0/1 before log
                ŷ_safe = ŷ_prob.clamp(_PROB_EPS, 1.0 - _PROB_EPS)
                bce_raw = F.binary_cross_entropy(ŷ_safe, y, reduction="none")
                bce_raw = torch.where(y > 0.5, pw * bce_raw, bce_raw)
                bce     = bce_raw.mean()

            bce_terms.append(w * bce)

            # ── False-positive penalty ───────────────────────────────────────────
            ŷ_safe = ŷ_prob.clamp(0.0, 1.0 - _PROB_EPS)
            fp     = ((1.0 - y) * ŷ_safe).mean()
            fp_terms.append(fp)

            # ── HI-critical FP penalty ───────────────────────────────────────────
            hi_fp  = ((1.0 - y) * ŷ_safe * hi_critical).mean()
            hi_fp_terms.append(hi_fp)

        bce_total   = sum(bce_terms)
        fp_total    = self.lam * sum(fp_terms)    / len(self.scheduler_names)
        hi_fp_total = self.mu  * sum(hi_fp_terms) / len(self.scheduler_names)
        total       = bce_total + fp_total + hi_fp_total

        return {
            "total":          total,
            "bce":            bce_total,
            "fp_penalty":     fp_total,
            "hi_fp_penalty":  hi_fp_total,
            "bce_per_sched":  torch.stack([t.detach() for t in bce_terms]),
        }