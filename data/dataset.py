"""
IMC-Former Dataset
==================
Loads IMC task-set CSVs and produces per-task feature tensors with padding masks.

Dataset CSV format (example for n=4, generalizes to any n):
  Taskset_ID, Num_Tasks, U_total, HI_ratio, CF,
  CLO1, CHI1, D1, T1, Crit1,
  CLO2, CHI2, D2, T2, Crit2,
  ...
  CLOn, CHIn, Dn, Tn, Critn,
  AMC_IMC_Schedulable, TT_Merge_Schedulable, IMC_PnG_Schedulable, EDF_IMC_Schedulable

Per-task feature vector f_i ∈ ℝ^12:
  [CLO_i, CHI_i, T_i, D_i, chi_i,
   U_LO_i, U_HI_i, rho_LO_i, rho_HI_i, delta_i, Delta_i, gamma_i]

Where:
  U_LO_i   = CLO_i / T_i               (LO utilization)
  U_HI_i   = CHI_i / T_i               (HI utilization; 0 for LO tasks)
  rho_LO_i = CLO_i / min(D_i, T_i)    (LO density)
  rho_HI_i = CHI_i / min(D_i, T_i)    (HI density; 0 for LO tasks)
  delta_i  = D_i / T_i                 (deadline ratio)
  Delta_i  = CHI_i - CLO_i             (optional execution budget)
  gamma_i  = CHI_i / CLO_i if CLO_i>0 else 0  (per-task criticality inflation)

Set-level context features z_set ∈ ℝ^6:
  [U_tot_LO, U_tot_HI, HI_ratio, CF_global, slack_margin, HI_demand]

Labels y ∈ {0,1}^4:
  [AMC_IMC, TT_Merge, IMC_PnG, EDF_IMC]
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---- Column name constants ----
SET_COLS = ["Taskset_ID", "Num_Tasks", "U_total", "HI_ratio", "CF"]
LABEL_COLS = [
    "AMC_IMC_Schedulable",
    "TT_Merge_Schedulable",
    "IMC_PnG_Schedulable",
    "EDF_IMC_Schedulable",
]
TASK_FEATURE_DIM = 12
CONTEXT_FEATURE_DIM = 6


def _parse_task_columns(df: pd.DataFrame) -> int:
    """
    Infer max_n from column names of the form CLO1, CHI1, D1, T1, Crit1, ...
    Returns the maximum task index found.
    """
    clo_cols = [c for c in df.columns if re.match(r"^CLO\d+$", c)]
    if not clo_cols:
        raise ValueError("No task columns (CLO1, CLO2, ...) found in CSV.")
    indices = [int(re.search(r"\d+", c).group()) for c in clo_cols]
    return max(indices)


def _extract_per_task_features(
    row: pd.Series, n: int, max_n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix F ∈ ℝ^{max_n × 12} and mask M ∈ {0,1}^{max_n} for one row.

    Real tasks occupy indices 0..n-1; padding occupies n..max_n-1.
    Mask M[i] = 1 for real tasks, 0 for padding.

    Per-task features:
      idx 0: CLO_i
      idx 1: CHI_i
      idx 2: T_i
      idx 3: D_i
      idx 4: chi_i  (criticality level)
      idx 5: U_LO_i = CLO_i / T_i
      idx 6: U_HI_i = CHI_i / T_i
      idx 7: rho_LO_i = CLO_i / min(D_i, T_i)
      idx 8: rho_HI_i = CHI_i / min(D_i, T_i)
      idx 9: delta_i = D_i / T_i
      idx 10: Delta_i = CHI_i - CLO_i
      idx 11: gamma_i = CHI_i / CLO_i (or 0 if CLO_i == 0)
    """
    F = np.zeros((max_n, TASK_FEATURE_DIM), dtype=np.float32)
    M = np.zeros(max_n, dtype=np.float32)

    for i in range(1, n + 1):
        clo = float(row.get(f"CLO{i}", 0))
        chi = float(row.get(f"CHI{i}", 0))
        T   = float(row.get(f"T{i}", 1))
        D   = float(row.get(f"D{i}", T))
        xi  = float(row.get(f"Crit{i}", 0))

        # Safety: avoid division by zero
        T_safe = max(T, 1e-6)
        D_safe = max(D, 1e-6)
        min_DT = max(min(D, T), 1e-6)

        U_LO   = clo / T_safe
        U_HI   = chi / T_safe
        rho_LO = clo / min_DT
        rho_HI = chi / min_DT
        delta  = D_safe / T_safe
        Delta  = chi - clo
        gamma  = (chi / clo) if clo > 1e-6 else 0.0

        idx = i - 1  # 0-indexed
        F[idx] = [clo, chi, T, D, xi, U_LO, U_HI, rho_LO, rho_HI, delta, Delta, gamma]
        M[idx] = 1.0

    return F, M


def _extract_context_features(row: pd.Series, F: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Build set-level context vector z ∈ ℝ^6.

    Features:
      0: U_tot_LO     = Σ_i (U_LO_i * M_i)
      1: U_tot_HI     = Σ_i (U_HI_i * M_i * chi_i)
      2: HI_ratio     = from CSV (or recomputed)
      3: CF_global    = from CSV
      4: slack_margin = 1 - U_tot_LO
      5: HI_demand    = Σ_i (U_HI_i * M_i * chi_i)  [same as U_tot_HI for now]
    """
    mask = M.astype(bool)
    U_LO_col  = F[:, 5]   # U_LO per task
    U_HI_col  = F[:, 6]   # U_HI per task
    chi_col   = F[:, 4]   # criticality level

    U_tot_LO  = float(np.sum(U_LO_col[mask]))
    HI_mask   = mask & (chi_col > 0.5)
    U_tot_HI  = float(np.sum(U_HI_col[HI_mask])) if HI_mask.any() else 0.0

    HI_ratio   = float(row.get("HI_ratio", 0.0))
    CF_global  = float(row.get("CF", 1.0))
    slack      = max(1.0 - U_tot_LO, 0.0)
    HI_demand  = U_tot_HI  # same quantity, kept separate for potential future divergence

    return np.array([U_tot_LO, U_tot_HI, HI_ratio, CF_global, slack, HI_demand],
                    dtype=np.float32)


class IMCDataset(Dataset):
    """
    PyTorch Dataset for IMC task sets.

    Each item is a dict:
      "features"  : FloatTensor (max_n, 12) — per-task features, padded
      "mask"      : FloatTensor (max_n,)    — 1 for real tasks, 0 for padding
      "context"   : FloatTensor (6,)        — set-level context
      "labels"    : FloatTensor (4,)        — binary schedulability labels
      "n_tasks"   : int                     — actual number of tasks
      "taskset_id": int                     — dataset row identifier

    Args:
        csv_path   : path to CSV file
        max_n      : maximum task count across ALL files in the experiment;
                     all feature tensors are padded to this size.
                     If None, inferred from this file (use with care for multi-file experiments).
        normalize  : if True, apply per-feature normalization using stats computed on this split.
                     Pass precomputed stats via norm_stats to apply train-set statistics to val/test.
        norm_stats : dict with keys 'mean' and 'std', each FloatTensor (12,).
                     If None and normalize=True, computes from this split.
    """

    def __init__(
        self,
        csv_path: str,
        max_n: Optional[int] = None,
        normalize: bool = True,
        norm_stats: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.csv_path = csv_path

        df = pd.read_csv(csv_path)
        self._validate_columns(df)

        inferred_max = _parse_task_columns(df)
        self.max_n = max_n if max_n is not None else inferred_max

        # Build raw arrays
        all_features = []
        all_masks    = []
        all_context  = []
        all_labels   = []
        all_n        = []
        all_ids      = []

        for _, row in df.iterrows():
            n = int(row["Num_Tasks"])
            F, M = _extract_per_task_features(row, n, self.max_n)
            z    = _extract_context_features(row, F, M)
            y    = np.array([float(row[c]) for c in LABEL_COLS], dtype=np.float32)

            all_features.append(F)
            all_masks.append(M)
            all_context.append(z)
            all_labels.append(y)
            all_n.append(n)
            all_ids.append(int(row.get("Taskset_ID", 0)))

        self.features = torch.from_numpy(np.stack(all_features))  # (N, max_n, 12)
        self.masks    = torch.from_numpy(np.stack(all_masks))     # (N, max_n)
        self.context  = torch.from_numpy(np.stack(all_context))   # (N, 6)
        self.labels   = torch.from_numpy(np.stack(all_labels))    # (N, 4)
        self.n_tasks  = all_n
        self.ids      = all_ids

        # Normalization: computed on real (non-padded) task features only
        self.normalize  = normalize
        self.norm_stats = norm_stats
        if normalize:
            if norm_stats is None:
                self.norm_stats = self._compute_norm_stats()
            self._apply_normalization()

    def _validate_columns(self, df: pd.DataFrame):
        missing_labels = [c for c in LABEL_COLS if c not in df.columns]
        if missing_labels:
            raise ValueError(f"Missing label columns: {missing_labels}")
        if "Num_Tasks" not in df.columns:
            raise ValueError("CSV must contain 'Num_Tasks' column.")

    def _compute_norm_stats(self) -> Dict[str, torch.Tensor]:
        """
        Compute mean and std of per-task features over all real (non-padded) tasks.
        Shape of output tensors: (12,)
        """
        mask_expanded = self.masks.unsqueeze(-1).expand_as(self.features)  # (N, max_n, 12)
        # Gather only real-task feature vectors
        real_features = self.features[mask_expanded.bool()].reshape(-1, TASK_FEATURE_DIM)
        mean = real_features.mean(dim=0)
        std  = real_features.std(dim=0).clamp(min=1e-6)
        return {"mean": mean, "std": std}

    def _apply_normalization(self):
        """Apply z-score normalization to real task features; leave padding as zeros."""
        mean = self.norm_stats["mean"]  # (12,)
        std  = self.norm_stats["std"]   # (12,)

        # Normalize all positions, then zero out padded positions
        # features: (N, max_n, 12)
        self.features = (self.features - mean) / std
        # Zero out padding positions so they don't contribute to attention/pooling
        pad_mask = (1 - self.masks).bool().unsqueeze(-1).expand_as(self.features)
        self.features[pad_mask] = 0.0

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "features":   self.features[idx],   # (max_n, 12)
            "mask":       self.masks[idx],       # (max_n,)
            "context":    self.context[idx],     # (6,)
            "labels":     self.labels[idx],      # (4,)
            "n_tasks":    self.n_tasks[idx],     # int
            "taskset_id": self.ids[idx],         # int
        }

    def get_scheduler_class_weights(self) -> torch.Tensor:
        """
        Compute per-scheduler positive class weights for weighted BCE.
        Returns tensor of shape (4,) where weight_s = n_neg_s / n_pos_s.
        """
        weights = []
        for i in range(4):
            pos = self.labels[:, i].sum().item()
            neg = len(self.labels) - pos
            w   = neg / max(pos, 1)
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    cfg,
    drop_last_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[DataLoader]]:
    """
    Build all DataLoaders from config.

    Returns:
        train_loader     : DataLoader for training
        val_loader       : DataLoader for in-distribution validation
        test_loader      : DataLoader for in-distribution test
        gen_loaders      : list of DataLoaders for generalization splits
    """
    tcfg = cfg.training

    # ---- Training dataset (compute norm stats here, propagate to others) ----
    train_path = os.path.join(tcfg.data_dir, tcfg.train_file)
    train_ds = IMCDataset(train_path, normalize=True, norm_stats=None)
    norm_stats = train_ds.norm_stats

    # ---- Determine global max_n (across all files) ----
    # We do this to ensure all datasets use the same padding size.
    # In practice: set max_n to the largest n in generalization files.
    # This is handled by the caller; for simplicity we use train's max_n unless overridden.
    global_max_n = train_ds.max_n  # will be updated if gen files have larger n

    # ---- Curriculum sampling weights for training ----
    train_sample_weights = _curriculum_weights(
        train_ds, epoch=0,
        phase1_end=tcfg.phase1_end_epoch,
        phase2_end=tcfg.phase2_end_epoch,
    )
    sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg.batch_size,
        sampler=sampler,
        num_workers=tcfg.num_workers,
        pin_memory=True,
        drop_last=drop_last_train,
        collate_fn=_collate_fn,
    )

    # ---- Validation ----
    val_path = os.path.join(tcfg.data_dir, tcfg.val_file)
    val_ds = IMCDataset(
        val_path,
        max_n=global_max_n,
        normalize=True,
        norm_stats=norm_stats,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg.batch_size * 2,
        shuffle=False,
        num_workers=tcfg.num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    # ---- In-distribution test ----
    test_path = os.path.join(tcfg.data_dir, tcfg.test_in_dist_file)
    test_ds = IMCDataset(
        test_path,
        max_n=global_max_n,
        normalize=True,
        norm_stats=norm_stats,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=tcfg.batch_size * 2,
        shuffle=False,
        num_workers=tcfg.num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    # ---- Generalization splits ----
    gen_loaders = []
    for gf in tcfg.gen_files:
        gpath = os.path.join(tcfg.data_dir, gf)
        if not os.path.exists(gpath):
            gen_loaders.append(None)
            continue
        gds = IMCDataset(
            gpath,
            normalize=True,
            norm_stats=norm_stats,
        )
        gen_loaders.append(DataLoader(
            gds,
            batch_size=tcfg.batch_size * 2,
            shuffle=False,
            num_workers=tcfg.num_workers,
            pin_memory=True,
            collate_fn=_collate_fn,
        ))

    return train_loader, val_loader, test_loader, gen_loaders, norm_stats


def update_train_sampler(train_loader: DataLoader, epoch: int, cfg) -> DataLoader:
    """
    Rebuild the training DataLoader's sampler for the current curriculum phase.
    Call this at the start of each epoch.
    """
    tcfg = cfg.training
    train_ds = train_loader.dataset
    weights = _curriculum_weights(
        train_ds, epoch,
        phase1_end=tcfg.phase1_end_epoch,
        phase2_end=tcfg.phase2_end_epoch,
    )
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_ds),
        replacement=True,
    )
    return DataLoader(
        train_ds,
        batch_size=tcfg.batch_size,
        sampler=sampler,
        num_workers=tcfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_fn,
    )


def _curriculum_weights(
    dataset: IMCDataset,
    epoch: int,
    phase1_end: int = 20,
    phase2_end: int = 60,
) -> torch.Tensor:
    """
    Assign sampling weight to each training example based on its n_tasks
    and the current curriculum phase.

    Phase 1 (epochs < phase1_end):
        n in {4, 8}:       weight 0.40 each  (total 0.80)
        n in {12, 16, 20}: weight 0.067 each (total 0.20)

    Phase 2 (phase1_end <= epoch < phase2_end):
        Uniform weight across all n.

    Phase 3 (epoch >= phase2_end):
        n in {4, 8}:       weight 0.10 each
        n in {12}:         weight 0.20
        n in {16, 20}:     weight 0.30 each
        (Emphasize boundary of training distribution for fine-tuning)
    """
    n_arr = torch.tensor(dataset.n_tasks, dtype=torch.float32)
    weights = torch.ones(len(dataset), dtype=torch.float32)

    if epoch < phase1_end:
        weights[n_arr <= 8]  = 2.0
        weights[n_arr > 8]   = 0.4
    elif epoch < phase2_end:
        weights[:] = 1.0  # uniform
    else:
        weights[n_arr <= 8]                    = 0.5
        weights[(n_arr > 8) & (n_arr <= 12)]   = 1.0
        weights[(n_arr > 12) & (n_arr <= 20)]  = 2.0

    return weights


def _collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate: stack all tensors, keep n_tasks and taskset_id as lists.
    Handles variable-length padding — each item already has the same max_n
    (padding was done in __getitem__), so standard stacking works.
    """
    return {
        "features":   torch.stack([b["features"]   for b in batch]),   # (B, max_n, 12)
        "mask":       torch.stack([b["mask"]        for b in batch]),   # (B, max_n)
        "context":    torch.stack([b["context"]     for b in batch]),   # (B, 6)
        "labels":     torch.stack([b["labels"]      for b in batch]),   # (B, 4)
        "n_tasks":    [b["n_tasks"]    for b in batch],
        "taskset_ids":[b["taskset_id"] for b in batch],
    }
