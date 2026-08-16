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
  is most dangerous: the HI task may miss its deadline under actual execution,
  which can be catastrophic. Controlled by μ (hi_fp_penalty_mu).

  A task set is "HI-critical" if: max_{HI tasks} U_HI_i > hi_critical_threshold.
  This is computed from the raw feature matrix (feature index 6 = U_HI_i).

Note: No consistency regularization across schedulers is included (pending
mathematical verification of the partial ordering between schedulers).
Note: No utilization-margin focal loss is included (pending justification).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from configs.config import LossConfig


class IMCLoss(nn.Module):
    """
    Safety-aware loss for IMC schedulability prediction.

    Args:
        cfg             : LossConfig with λ, μ, thresholds, and scheduler weights
        scheduler_names : list of scheduler name strings (must match model head names)
        pos_weights     : optional per-scheduler positive class weights (B_neg/B_pos ratio)
                          for handling dataset imbalance in BCE. Shape (4,) or None.
    """

    def __init__(
        self,
        cfg: LossConfig,
        scheduler_names: List[str],
        pos_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.scheduler_names = scheduler_names
        n_sched = len(scheduler_names)

        # Per-scheduler BCE weights (from config)
        self.register_buffer(
            "sched_weights",
            torch.tensor(cfg.scheduler_weights, dtype=torch.float32),
        )

        # Optional positive class weights for imbalanced datasets
        # BCE with pos_weight: loss = -[pos_w * y * log(ŷ) + (1-y) * log(1-ŷ)]
        if pos_weights is not None:
            self.register_buffer("pos_weights", pos_weights)  # (n_sched,)
        else:
            self.register_buffer("pos_weights", torch.ones(n_sched))

        self.lam = cfg.fp_penalty_lambda
        self.mu  = cfg.hi_fp_penalty_mu
        self.hi_thresh = cfg.hi_critical_threshold

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the safety-aware IMC loss.

        Args:
            logits  : dict {scheduler_name: (B,) probabilities in [0,1]}
            labels  : (B, 4) float tensor of ground-truth schedulability labels
            features: (B, N, 12) per-task features — used to identify HI-critical sets
            mask    : (B, N) float — 1=real task, 0=padding

        Returns:
            dict with keys:
              "total"       : scalar total loss
              "bce"         : scalar — summed weighted BCE across schedulers
              "fp_penalty"  : scalar — FP penalty term
              "hi_fp_penalty": scalar — HI-critical FP penalty term
              "bce_per_sched": (4,) — per-scheduler BCE values
        """
        B = labels.shape[0]
        bce_terms   = []
        fp_terms    = []
        hi_fp_terms = []

        # ---- Identify HI-critical task sets ----
        # Feature index 6 = U_HI_i; feature index 4 = chi_i (criticality)
        # After normalisation chi_i > 0 → HI task
        chi_norm = features[:, :, 4]          # (B, N) — normalised criticality
        u_hi     = features[:, :, 6]          # (B, N) — normalised U_HI
        real     = mask > 0.5                  # (B, N) bool

        # Only consider HI tasks (chi_norm > 0 after normalisation means original chi=1)
        hi_tasks = (chi_norm > 0) & real       # (B, N) bool
        # Max U_HI among real HI tasks; 0 if no HI tasks
        u_hi_max = (u_hi * hi_tasks.float()).max(dim=1).values  # (B,)
        hi_critical = (u_hi_max > self.hi_thresh).float()       # (B,) — 1 if HI-critical

        # ---- Per-scheduler loss terms ----
        for s_idx, name in enumerate(self.scheduler_names):
            ŷ = logits[name]           # (B,) probabilities
            y = labels[:, s_idx]       # (B,) ground-truth {0, 1}
            w = self.sched_weights[s_idx]
            pw = self.pos_weights[s_idx]

            # ---- BCE with positive class weight ----
            bce = F.binary_cross_entropy(
                ŷ, y,
                weight=None,
                reduction="none",
            )  # (B,)
            # Apply positive class weight manually (cannot use pos_weight with probabilities)
            bce_weighted = torch.where(y > 0.5, pw * bce, bce).mean()
            bce_terms.append(w * bce_weighted)

            # ---- False-positive penalty ----
            # (1 - y) * ŷ:
            #   y=0 (unschedulable): contributes ŷ — penalise high predicted schedulability
            #   y=1 (schedulable):   contributes 0 — no penalty
            fp = ((1.0 - y) * ŷ).mean()
            fp_terms.append(fp)

            # ---- HI-critical FP penalty ----
            # Same as FP penalty but only on HI-critical task sets
            hi_fp = ((1.0 - y) * ŷ * hi_critical).mean()
            hi_fp_terms.append(hi_fp)

        # ---- Aggregate ----
        bce_total    = sum(bce_terms)
        fp_total     = self.lam * sum(fp_terms)   / len(self.scheduler_names)
        hi_fp_total  = self.mu  * sum(hi_fp_terms) / len(self.scheduler_names)

        total = bce_total + fp_total + hi_fp_total

        return {
            "total":          total,
            "bce":            bce_total,
            "fp_penalty":     fp_total,
            "hi_fp_penalty":  hi_fp_total,
            "bce_per_sched":  torch.stack([t.detach() for t in bce_terms]),
        }
