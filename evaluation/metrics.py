"""
IMC-Former Evaluation Metrics
==============================
All metrics are computed per-scheduler and aggregated.

Primary metric: FPR (False Positive Rate) = FP / (FP + TN)
  — a model that predicts schedulable on a truly unschedulable system
    can cause deadline misses; this is the safety-critical metric.

Secondary metrics (in order of importance):
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1 Score  = 2 * Precision * Recall / (Precision + Recall)
  Accuracy  = (TP + TN) / Total
  AUROC     — threshold-independent, measures ranking quality

All metrics are reported:
  - Per scheduler head (AMC_IMC, TT_Merge, IMC_PnG, EDF_IMC)
  - Averaged across schedulers (macro average)
  - Broken down by task-set size n (for generalization analysis)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    scheduler_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute all metrics for a batch or full split.

    Args:
        probs           : (N, S) float — predicted schedulability probabilities
        labels          : (N, S) float — ground-truth binary labels
        threshold       : classification threshold (default 0.5)
        scheduler_names : list of S scheduler names; defaults to ["S0","S1",...]

    Returns:
        flat dict of {metric_scheduler: value, ..., metric_avg: value}
    """
    N, S = probs.shape
    if scheduler_names is None:
        scheduler_names = [f"S{i}" for i in range(S)]

    probs_np  = probs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    preds_np  = (probs_np >= threshold).astype(int)

    results = {}
    fpr_list, prec_list, rec_list, f1_list, acc_list, auc_list = [], [], [], [], [], []

    for i, name in enumerate(scheduler_names):
        y     = labels_np[:, i]
        ŷ_p   = probs_np[:, i]
        ŷ     = preds_np[:, i]

        TP = int(((ŷ == 1) & (y == 1)).sum())
        TN = int(((ŷ == 0) & (y == 0)).sum())
        FP = int(((ŷ == 1) & (y == 0)).sum())
        FN = int(((ŷ == 0) & (y == 1)).sum())

        fpr   = FP / max(FP + TN, 1)
        prec  = TP / max(TP + FP, 1)
        rec   = TP / max(TP + FN, 1)
        f1    = 2 * prec * rec / max(prec + rec, 1e-6)
        acc   = (TP + TN) / max(N, 1)

        # AUROC (handle degenerate case where all labels are same class)
        try:
            auc = roc_auc_score(y, ŷ_p)
        except ValueError:
            auc = float("nan")

        results[f"fpr/{name}"]       = fpr
        results[f"precision/{name}"] = prec
        results[f"recall/{name}"]    = rec
        results[f"f1/{name}"]        = f1
        results[f"accuracy/{name}"]  = acc
        results[f"auroc/{name}"]     = auc
        results[f"tp/{name}"]        = TP
        results[f"fp/{name}"]        = FP
        results[f"tn/{name}"]        = TN
        results[f"fn/{name}"]        = FN

        fpr_list.append(fpr)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        acc_list.append(acc)
        auc_list.append(auc)

    # Macro averages
    results["fpr/avg"]       = float(np.nanmean(fpr_list))
    results["precision/avg"] = float(np.nanmean(prec_list))
    results["recall/avg"]    = float(np.nanmean(rec_list))
    results["f1/avg"]        = float(np.nanmean(f1_list))
    results["accuracy/avg"]  = float(np.nanmean(acc_list))
    results["auroc/avg"]     = float(np.nanmean(auc_list))

    return results


def compute_metrics_by_n(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_tasks: List[int],
    scheduler_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[int, Dict[str, float]]:
    """
    Compute metrics broken down by task-set size n.
    Used to analyse generalization behaviour.

    Args:
        probs          : (N_total, S)
        labels         : (N_total, S)
        n_tasks        : list of n for each example
        scheduler_names: S scheduler names
        threshold      : classification threshold

    Returns:
        dict: {n_value: {metric: float}}
    """
    n_arr     = torch.tensor(n_tasks)
    unique_ns = sorted(n_arr.unique().tolist())

    results_by_n = {}
    for n_val in unique_ns:
        idx = (n_arr == n_val)
        if idx.sum() == 0:
            continue
        results_by_n[int(n_val)] = compute_metrics(
            probs[idx], labels[idx], threshold, scheduler_names
        )
    return results_by_n


def accumulate_predictions(
    all_probs: List[torch.Tensor],
    all_labels: List[torch.Tensor],
    all_n: List[List[int]],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Concatenate prediction buffers accumulated during evaluation loop.

    Args:
        all_probs  : list of (B_i, S) tensors
        all_labels : list of (B_i, S) tensors
        all_n      : list of lists of int

    Returns:
        probs_cat  : (N_total, S)
        labels_cat : (N_total, S)
        n_cat      : list of N_total ints
    """
    probs_cat  = torch.cat(all_probs,  dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    n_cat      = [n for sublist in all_n for n in sublist]
    return probs_cat, labels_cat, n_cat
