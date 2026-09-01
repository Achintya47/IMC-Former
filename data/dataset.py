"""
IMC-Former Dataset
==================
Loads IMC task-set CSVs and produces per-task feature tensors with padding masks.

BUG-FIXED (v3, memory): the previous loader built every row's feature tensor
via a Python-level `df.iterrows()` loop (millions of Python-object allocations
for a multi-million-row corpus), accumulated per-row numpy arrays into a
Python list, then called `np.stack()` on the whole list (which allocates a
SECOND full-size array before the first can be freed -- a transient 2x memory
spike on top of the already-loaded pandas DataFrame). Combined with
`pd.read_csv`'s default float64/int64 dtypes (2x the memory of float32/int32
for purely numeric columns) and eagerly loading train + val + test + every
generalization split's full IMCDataset before training even starts, this
reliably exhausted host RAM once the prepared corpus grew past ~1-2M rows
(e.g. after switching to a joint-distribution-preserving data prep step that,
unlike the old marginal-balancing LP, does not aggressively downsample).

This version:
  1. Reads each CSV with explicit, memory-minimal dtypes.
  2. Extracts per-task features with vectorized numpy operations grouped by
     Num_Tasks (a handful of groups, e.g. n in {4,8,12,16,20}), instead of a
     Python loop over every individual row.
  3. Writes directly into a single pre-allocated (N, max_n, 12) array instead
     of stacking a list of per-row arrays.
  4. Frees the source DataFrame as soon as extraction is done.
  5. Supports lazy construction of generalization-split datasets (see
     `build_dataloaders(..., eager_gen=False)` and `LazyGenSplit` below), so
     memory for gen_small..gen_xxlarge is only paid at evaluation time, not
     throughout the entire training run.

All formulas and semantics are identical to the original row-by-row version;
only the mechanics of computing them changed. See test_dataset_equivalence.py
for a row-by-row numerical equivalence check against the original
implementation.
"""

import gc
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
    clo_cols = [c for c in df.columns if re.match(r"^CLO\d+$", c)]
    if not clo_cols:
        raise ValueError("No task columns (CLO1, CLO2, ...) found in CSV.")
    indices = [int(re.search(r"\d+", c).group()) for c in clo_cols]
    return max(indices)


def _build_dtype_map(columns: List[str]) -> Dict[str, str]:
    """
    Explicit, memory-minimal dtypes for pd.read_csv.

    Per-task columns (CLO{i}, CHI{i}, D{i}, T{i}, Crit{i}) MUST stay float
    (not int) even though their values are conceptually integers/{0,1}: when
    a prepared CSV mixes multiple task-set sizes (e.g. n=4 and n=20 rows in
    the same train.csv), pandas fills a smaller row's higher-index task
    columns with NaN once everything is concatenated to a common column set,
    and integer dtypes cannot represent NaN. float32 is still half the memory
    of the pandas default float64.
    """
    dtype_map: Dict[str, str] = {}
    for c in columns:
        if c == "Taskset_ID":
            dtype_map[c] = "int32"
        elif c == "Num_Tasks":
            dtype_map[c] = "int16"
        elif c in ("U_total", "HI_ratio", "CF"):
            dtype_map[c] = "float32"
        elif re.match(r"^(CLO|CHI|D|T|Crit)\d+$", c):
            dtype_map[c] = "float32"
        elif c in LABEL_COLS:
            dtype_map[c] = "float32"  # kept float so a NaN label fails the
                                       # finiteness check below instead of
                                       # silently raising inside read_csv
    return dtype_map


def _extract_features_vectorized(
    df: pd.DataFrame, max_n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized replacement for the old per-row _extract_per_task_features loop.

    Rows are grouped by their own Num_Tasks (there are only a handful of
    distinct values in practice -- e.g. {4,8,12,16,20} -- not one group per
    row), and each group's task columns are extracted and transformed with a
    single batch of numpy array operations instead of a Python-level loop.

    Returns:
        features : (N, max_n, 12) float32, zero-padded beyond each row's n
        mask     : (N, max_n) float32, 1 for real tasks, 0 for padding
        num_tasks: (N,) int32
    """
    N = len(df)
    features = np.zeros((N, max_n, TASK_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((N, max_n), dtype=np.float32)

    num_tasks = df["Num_Tasks"].to_numpy(dtype=np.int32, copy=True)

    for n in np.unique(num_tasks):
        n = int(n)
        if n <= 0:
            continue
        rows_idx = np.flatnonzero(num_tasks == n)
        if len(rows_idx) == 0:
            continue

        clo_cols = [f"CLO{i}" for i in range(1, n + 1)]
        chi_cols = [f"CHI{i}" for i in range(1, n + 1)]
        T_cols   = [f"T{i}"   for i in range(1, n + 1)]
        D_cols   = [f"D{i}"   for i in range(1, n + 1)]
        xi_cols  = [f"Crit{i}" for i in range(1, n + 1)]

        clo = df[clo_cols].to_numpy(dtype=np.float64)[rows_idx]  # (n_rows, n)
        chi = df[chi_cols].to_numpy(dtype=np.float64)[rows_idx]
        T   = df[T_cols].to_numpy(dtype=np.float64)[rows_idx]
        D   = df[D_cols].to_numpy(dtype=np.float64)[rows_idx]
        xi  = df[xi_cols].to_numpy(dtype=np.float64)[rows_idx]

        T_safe = np.maximum(T, 1e-6)
        D_safe = np.maximum(D, 1e-6)
        min_DT = np.maximum(np.minimum(D, T), 1e-6)

        U_LO   = clo / T_safe
        U_HI   = chi / T_safe
        rho_LO = clo / min_DT
        rho_HI = chi / min_DT
        delta  = D_safe / T_safe
        Delta  = chi - clo
        gamma  = np.divide(
            chi, clo,
            out=np.zeros_like(chi),
            where=(clo > 1e-6),
        )

        # (n_rows, n, 12) in the same feature order as the original code.
        block = np.stack(
            [clo, chi, T, D, xi, U_LO, U_HI, rho_LO, rho_HI, delta, Delta, gamma],
            axis=-1,
        ).astype(np.float32)

        features[rows_idx, :n, :] = block
        mask[rows_idx, :n] = 1.0

    return features, mask, num_tasks


def _extract_context_vectorized(
    df: pd.DataFrame, features: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Vectorized replacement for the old per-row _extract_context_features loop."""
    N = features.shape[0]
    mask_bool = mask.astype(bool)

    U_LO_col = features[:, :, 5]
    U_HI_col = features[:, :, 6]
    chi_col  = features[:, :, 4]

    U_tot_LO = (U_LO_col * mask).sum(axis=1)

    HI_mask  = mask_bool & (chi_col > 0.5)
    has_hi   = HI_mask.any(axis=1)
    U_tot_HI = np.where(has_hi, (U_HI_col * HI_mask).sum(axis=1), 0.0)

    if "HI_ratio" in df.columns:
        HI_ratio = df["HI_ratio"].to_numpy(dtype=np.float32, copy=True)
    else:
        HI_ratio = np.zeros(N, dtype=np.float32)

    if "CF" in df.columns:
        CF_global = df["CF"].to_numpy(dtype=np.float32, copy=True)
    else:
        CF_global = np.ones(N, dtype=np.float32)

    slack     = np.clip(1.0 - U_tot_LO, 0.0, None)
    HI_demand = U_tot_HI

    context = np.stack(
        [U_tot_LO, U_tot_HI, HI_ratio, CF_global, slack, HI_demand], axis=-1
    ).astype(np.float32)

    return context


class IMCDataset(Dataset):
    """
    PyTorch Dataset for IMC task sets.

    Every raw/derived tensor is validated for non-finite (NaN/Inf) values
    immediately after construction, BEFORE normalization and before anything
    is handed to the model. A single malformed CSV field (blank period,
    corrupted deadline, etc.) previously propagated silently all the way to a
    training step -- potentially many epochs later, whenever the
    WeightedRandomSampler happened to draw that particular row -- at which
    point it destroyed the entire model via a single NaN gradient. Failing
    loudly at data-load time, with the exact Taskset_ID, turns a multi-epoch
    mystery into an immediate, fixable error message.
    """

    def __init__(
        self,
        csv_path: str,
        max_n: Optional[int] = None,
        normalize: bool = True,
        norm_stats: Optional[Dict[str, torch.Tensor]] = None,
        validate_finite: bool = True,
    ):
        super().__init__()
        self.csv_path = csv_path

        # ── Memory-efficient read: explicit dtypes instead of pandas defaults ──
        header_df = pd.read_csv(csv_path, nrows=0)
        dtype_map = _build_dtype_map(list(header_df.columns))
        df = pd.read_csv(csv_path, dtype=dtype_map)

        self._validate_columns(df)

        inferred_max = _parse_task_columns(df)
        self.max_n = max_n if max_n is not None else inferred_max

        # ── Vectorized extraction (no per-row Python loop, no list+stack) ──
        features, mask, num_tasks = _extract_features_vectorized(df, self.max_n)
        context = _extract_context_vectorized(df, features, mask)
        labels  = df[LABEL_COLS].to_numpy(dtype=np.float32, copy=True)

        if "Taskset_ID" in df.columns:
            ids = df["Taskset_ID"].to_numpy(dtype=np.int64, copy=True).tolist()
        else:
            ids = list(range(len(df)))

        # Free the (potentially very large, wide, NaN-padded) DataFrame as
        # soon as we've pulled everything we need out of it. This matters:
        # for the duration of __init__ it previously coexisted in memory
        # alongside the full set of extracted tensors.
        del df, header_df
        gc.collect()

        self.features = torch.from_numpy(features)   # (N, max_n, 12) float32
        self.masks    = torch.from_numpy(mask)        # (N, max_n) float32
        self.context  = torch.from_numpy(context)     # (N, 6) float32
        self.labels   = torch.from_numpy(labels)       # (N, 4) float32
        self.n_tasks  = num_tasks.tolist()
        self.ids      = ids

        if validate_finite:
            self._validate_finite(csv_path)

        self.normalize  = normalize
        self.norm_stats = norm_stats
        if normalize:
            if norm_stats is None:
                self.norm_stats = self._compute_norm_stats()
            self._apply_normalization()

            if validate_finite:
                self._validate_finite(csv_path, stage="post-normalization",
                                      tensors={"features": self.features})

    def _validate_columns(self, df: pd.DataFrame):
        missing_labels = [c for c in LABEL_COLS if c not in df.columns]
        if missing_labels:
            raise ValueError(f"Missing label columns: {missing_labels}")
        if "Num_Tasks" not in df.columns:
            raise ValueError("CSV must contain 'Num_Tasks' column.")

    def _validate_finite(self, csv_path: str, stage: str = "raw",
                         tensors: Optional[Dict[str, torch.Tensor]] = None):
        """
        Raise a clear, actionable error identifying exactly which rows (by
        Taskset_ID) and which tensor contain NaN/Inf, instead of letting bad
        values silently reach the model.
        """
        if tensors is None:
            tensors = {
                "features": self.features,
                "context":  self.context,
                "labels":   self.labels,
            }

        for name, tensor in tensors.items():
            bad = ~torch.isfinite(tensor)
            if not bad.any():
                continue

            bad_rows = bad.reshape(bad.shape[0], -1).any(dim=1)
            bad_indices = bad_rows.nonzero(as_tuple=True)[0].tolist()
            bad_ids = [self.ids[i] for i in bad_indices[:20]]

            n_nan = torch.isnan(tensor).sum().item()
            n_inf = torch.isinf(tensor).sum().item()

            raise ValueError(
                f"\n{'='*78}\n"
                f"NON-FINITE VALUES DETECTED  [{stage}]\n"
                f"{'='*78}\n"
                f"  File          : {csv_path}\n"
                f"  Tensor        : '{name}'  shape={tuple(tensor.shape)}\n"
                f"  NaN count     : {n_nan}\n"
                f"  Inf count     : {n_inf}\n"
                f"  Affected rows : {len(bad_indices)} / {tensor.shape[0]}\n"
                f"  Taskset_ID(s) (first 20): {bad_ids}\n"
                f"{'='*78}\n"
                f"This almost certainly means a malformed field in the source CSV\n"
                f"(e.g. a blank/garbled period, deadline, or execution-time column\n"
                f"for one of the tasks in the affected row(s)). Training on this\n"
                f"data WILL eventually corrupt the model: a single NaN gradient\n"
                f"contaminates every parameter via gradient-norm clipping the\n"
                f"first time the WeightedRandomSampler happens to draw one of the\n"
                f"rows listed above, even if that doesn't happen until many epochs\n"
                f"in. Locate the listed Taskset_ID(s) in {csv_path} and fix or\n"
                f"remove them, then re-run.\n"
                f"{'='*78}\n"
            )

    def _compute_norm_stats(self) -> Dict[str, torch.Tensor]:
        mask_expanded = self.masks.unsqueeze(-1).expand_as(self.features)
        real_features = self.features[mask_expanded.bool()].reshape(-1, TASK_FEATURE_DIM)
        mean = real_features.mean(dim=0)
        std  = real_features.std(dim=0).clamp(min=1e-6)
        return {"mean": mean, "std": std}

    def _apply_normalization(self):
        mean = self.norm_stats["mean"]
        std  = self.norm_stats["std"]

        self.features = (self.features - mean) / std
        pad_mask = (1 - self.masks).bool().unsqueeze(-1).expand_as(self.features)
        self.features[pad_mask] = 0.0

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "features":   self.features[idx],
            "mask":       self.masks[idx],
            "context":    self.context[idx],
            "labels":     self.labels[idx],
            "n_tasks":    self.n_tasks[idx],
            "taskset_id": self.ids[idx],
        }

    def get_scheduler_class_weights(self) -> torch.Tensor:
        weights = []
        for i in range(4):
            pos = self.labels[:, i].sum().item()
            neg = len(self.labels) - pos
            w   = neg / max(pos, 1)
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)


class LazyGenSplit:
    """
    NEW: a placeholder for a generalization split that defers actually
    building the IMCDataset (and its DataLoader) until `.load()` is called.

    Generalization splits (gen_small, gen_medium, ..., gen_xxlarge) are only
    used once, at the very end of training, inside evaluate_generalization().
    The original code built all of them -- up to 5 additional full datasets,
    each its own copy of features/mask/context/labels tensors -- BEFORE
    training even started, so their memory was paid for the entire duration
    of training for no benefit. This class lets the trainer build one split,
    evaluate it, and free it before moving to the next, so peak memory is
    bounded by (train + val + test + ONE gen split) instead of
    (train + val + test + ALL gen splits).
    """

    def __init__(self, path: str, norm_stats: Dict[str, torch.Tensor], batch_size: int,
                 num_workers: int = 0):
        self.path = path
        self.norm_stats = norm_stats
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.exists = os.path.exists(path)

    def load(self) -> Optional[DataLoader]:
        if not self.exists:
            return None
        ds = IMCDataset(
            self.path,
            normalize=True,
            norm_stats=self.norm_stats,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_collate_fn,
        )


def build_dataloaders(
    cfg,
    drop_last_train: bool = True,
    eager_gen: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List, Dict]:
    """
    Build train/val/test DataLoaders (always eager -- needed throughout
    training) and generalization split handles.

    Args:
        eager_gen: if True, reproduces the ORIGINAL behavior of eagerly
            building every generalization split's IMCDataset up front (kept
            only for backward compatibility / debugging). Default False:
            generalization splits are returned as LazyGenSplit handles and
            are only actually loaded (one at a time) inside
            Trainer.evaluate_generalization(), which is the memory-safe path.

    Returns:
        train_loader, val_loader, test_loader, gen_items, norm_stats
        where gen_items is a list of LazyGenSplit (if eager_gen=False) or
        DataLoader/None (if eager_gen=True), matching cfg.training.gen_files
        order -- either way, downstream code should treat each item as
        "something .load()-able or already a DataLoader/None", see
        Trainer.evaluate_generalization().
    """
    tcfg = cfg.training

    train_path = os.path.join(tcfg.data_dir, tcfg.train_file)
    train_ds = IMCDataset(train_path, normalize=True, norm_stats=None)
    norm_stats = train_ds.norm_stats

    global_max_n = train_ds.max_n

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

    gen_items = []
    for gf in tcfg.gen_files:
        gpath = os.path.join(tcfg.data_dir, gf)
        if eager_gen:
            if not os.path.exists(gpath):
                gen_items.append(None)
                continue
            gds = IMCDataset(gpath, normalize=True, norm_stats=norm_stats)
            gen_items.append(DataLoader(
                gds, batch_size=tcfg.batch_size * 2, shuffle=False,
                num_workers=tcfg.num_workers, pin_memory=True,
                collate_fn=_collate_fn,
            ))
        else:
            gen_items.append(LazyGenSplit(
                gpath, norm_stats, batch_size=tcfg.batch_size * 2,
                num_workers=tcfg.num_workers,
            ))

    return train_loader, val_loader, test_loader, gen_items, norm_stats


def update_train_sampler(train_loader: DataLoader, epoch: int, cfg) -> DataLoader:
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
    n_arr = torch.tensor(dataset.n_tasks, dtype=torch.float32)
    weights = torch.ones(len(dataset), dtype=torch.float32)

    if epoch < phase1_end:
        weights[n_arr <= 8]  = 2.0
        weights[n_arr > 8]   = 0.4
    elif epoch < phase2_end:
        weights[:] = 1.0
    else:
        weights[n_arr <= 8]                    = 0.5
        weights[(n_arr > 8) & (n_arr <= 12)]   = 1.0
        weights[(n_arr > 12) & (n_arr <= 20)]  = 2.0

    return weights


def _collate_fn(batch: List[Dict]) -> Dict:
    return {
        "features":   torch.stack([b["features"]   for b in batch]),
        "mask":       torch.stack([b["mask"]        for b in batch]),
        "context":    torch.stack([b["context"]     for b in batch]),
        "labels":     torch.stack([b["labels"]      for b in batch]),
        "n_tasks":    [b["n_tasks"]    for b in batch],
        "taskset_ids":[b["taskset_id"] for b in batch],
    }