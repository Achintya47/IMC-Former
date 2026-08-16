"""
IMC-Former Configuration
========================
Central dataclass holding all hyperparameters and paths.
Modify this file (or pass overrides via CLI) rather than editing model/training code.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class ModelConfig:
    # ---- Per-task feature dimension (see feature_engineering.py) ----
    # f_i = [CLO, CHI, T, D, chi, U_LO, U_HI, rho_LO, rho_HI, delta, Delta, gamma]
    task_feature_dim: int = 12

    # ---- Encoder ----
    d_model: int = 256          # embedding dimension throughout model
    encoder_hidden: int = 512   # hidden dim of per-task MLP encoder
    encoder_layers: int = 2     # number of hidden layers in task encoder MLP

    # ---- Transformer (within each stream) ----
    n_heads: int = 8            # attention heads (must divide d_model)
    n_transformer_layers: int = 3  # transformer blocks per stream
    attn_dropout: float = 0.1
    ffn_dim: int = 512          # feedforward dim inside transformer blocks
    ffn_dropout: float = 0.1

    # ---- Cross-stream attention ----
    # Uses the same d_model; separate Q/K/V projections learned independently
    cross_attn_heads: int = 8
    cross_attn_dropout: float = 0.1

    # ---- Hierarchical pooling ----
    chunk_size: int = 8         # tasks per chunk in hierarchical attention pool
    pool_hidden: int = 256      # attention pool query/key dim

    # ---- Set-level context features ----
    # [U_tot_LO, U_tot_HI, HI_ratio, CF_global, slack_margin, HI_demand]
    context_feature_dim: int = 6
    context_hidden: int = 64   # MLP hidden dim for context projection

    # ---- Global fusion ----
    # c = Concat([c_LO, c_HI, z_set]) => dim = d_model + d_model + context_hidden
    # = 256 + 256 + 64 = 576
    fused_dim: int = 576  # auto-computed but stated explicitly for clarity

    # ---- Scheduler-specific prediction heads ----
    # 4 schedulers: AMC_IMC, TT_Merge, IMC_PnG, EDF_IMC
    scheduler_names: List[str] = field(default_factory=lambda: [
        "AMC_IMC", "TT_Merge", "IMC_PnG", "EDF_IMC"
    ])
    head_hidden: int = 256      # hidden dim of each prediction head MLP
    head_layers: int = 2        # number of hidden layers per head
    head_dropout: float = 0.1

    # ---- Padding ----
    max_tasks: int = 1024       # maximum task-set size supported (for buffer allocation)


@dataclass
class LossConfig:
    # Scheduler-specific BCE weights (equal by default; tune after first run)
    scheduler_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])

    # λ: strength of the false-positive penalty term
    # L_FP = λ * mean((1 - y) * ŷ)  over all schedulers and batch
    fp_penalty_lambda: float = 2.0

    # μ: ADDITIONAL multiplier for false positives on HI-critical task sets
    # A task set is "HI-critical" if any HI task has U_HI_i > hi_critical_threshold
    # These represent the most dangerous mispredictions
    hi_fp_penalty_mu: float = 3.0
    hi_critical_threshold: float = 0.7   # U_HI_i above this => HI-critical set


@dataclass
class TrainingConfig:
    # ---- Paths ----
    data_dir: str = "data"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    figure_dir: str = "figures"

    # ---- Data files (CSVs to be placed in data_dir) ----
    # Training and validation files should contain n ∈ {4,8,12,16,20}
    train_file: str = "train.csv"
    val_file: str = "val_in_dist.csv"
    test_in_dist_file: str = "test_in_dist.csv"

    # Generalization test files (n > 20, strictly held out)
    gen_files: List[str] = field(default_factory=lambda: [
        "gen_small.csv",    # n ∈ {24, 28}
        "gen_medium.csv",   # n ∈ {32, 40}
        "gen_large.csv",    # n ∈ {48, 64}
        "gen_xlarge.csv",   # n ∈ {128, 256}
        "gen_xxlarge.csv",  # n ∈ {512, 1000}
    ])
    gen_names: List[str] = field(default_factory=lambda: [
        "gen_small", "gen_medium", "gen_large", "gen_xlarge", "gen_xxlarge"
    ])

    # ---- Training ----
    num_epochs: int = 80
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42

    # ---- Optimizer ----
    encoder_lr: float = 3e-4    # learning rate for encoder + transformer layers
    head_lr: float = 1e-3       # learning rate for prediction heads
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # ---- LR Scheduler ----
    # ReduceLROnPlateau watching validation FPR (primary metric)
    lr_patience: int = 5
    lr_factor: float = 0.5
    lr_min: float = 1e-6

    # ---- Early stopping ----
    early_stop_patience: int = 10   # epochs without val FPR improvement
    min_accuracy_threshold: float = 0.80  # model must also clear this accuracy

    # ---- Curriculum learning phases ----
    # Phase 1: over-sample small n; Phase 2: uniform; Phase 3: over-sample large n
    phase1_end_epoch: int = 20
    phase2_end_epoch: int = 60
    # Phase 3 runs from epoch 60 to num_epochs

    # ---- Logging ----
    log_interval: int = 50          # batches between training log prints
    eval_interval: int = 1          # epochs between full validation evaluations
    save_best_only: bool = True
    run_name: str = "imc_former_base"

    # ---- Device ----
    device: str = "cuda"            # "cuda" or "cpu"


@dataclass
class IMCFormerConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self):
        """Sanity-check that dimensions are consistent."""
        assert self.model.d_model % self.model.n_heads == 0, \
            f"d_model ({self.model.d_model}) must be divisible by n_heads ({self.model.n_heads})"
        assert self.model.d_model % self.model.cross_attn_heads == 0, \
            f"d_model must be divisible by cross_attn_heads"
        expected_fused = self.model.d_model * 2 + self.model.context_hidden
        assert self.model.fused_dim == expected_fused, \
            f"fused_dim mismatch: expected {expected_fused}, got {self.model.fused_dim}"
        assert len(self.model.scheduler_names) == len(self.loss.scheduler_weights), \
            "scheduler_names and scheduler_weights must have the same length"
        return self


# ---- Default config (used when no overrides are provided) ----
DEFAULT_CONFIG = IMCFormerConfig().validate()
