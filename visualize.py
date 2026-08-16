"""
visualize.py — Research-paper figures for IMC-Former
=====================================================
Generates the major plots needed for a research paper:

  Fig 1. Training curves: Loss components over epochs
  Fig 2. FPR and Precision vs epoch (primary safety metrics)
  Fig 3. Precision–Recall by task-set size n (generalization)
  Fig 4. FPR by task-set size n across scheduling policies
  Fig 5. Per-scheduler FPR / Recall bar chart (in-dist test)
  Fig 6. ROC curves per scheduler (in-dist test)
  Fig 7. Confusion matrices (2×2) per scheduler

Usage:
    python visualize.py \\
        --history_json  logs/imc_former_base_history.json \\
        --gen_json      logs/imc_former_base_gen_results.json \\
        --output_dir    figures/
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SCHEDULER_NAMES = ["AMC_IMC", "TT_Merge", "IMC_PnG", "EDF_IMC"]
SCHED_COLORS    = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
SCHED_LABELS    = ["AMC-IMC", "TT-Merge", "IMC-PnG", "EDF-IMC"]

# Publication-quality style
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "lines.linewidth":  1.8,
})


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_history_series(history: list, key: str, split: str = "val") -> list:
    """Pull a metric time series from the history list."""
    return [ep[split].get(key, float("nan")) for ep in history if split in ep]


def savefig(fig, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 1: Training curves — loss components ────────────────────────────

def fig_loss_curves(history: list, output_dir: str):
    epochs = [ep["epoch"] for ep in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: total loss + components (train)
    ax = axes[0]
    for key, label, ls in [
        ("loss/total", "Total",      "-"),
        ("loss/bce",   "BCE",        "--"),
        ("loss/fp",    "FP penalty", "-."),
        ("loss/hi_fp", "HI-FP pen.", ":"),
    ]:
        vals = extract_history_series(history, key, "train")
        ax.plot(epochs, vals, ls=ls, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Components")
    ax.legend()

    # Right: val loss vs train loss
    ax = axes[1]
    train_loss = extract_history_series(history, "loss/total", "train")
    val_loss   = extract_history_series(history, "loss/total", "val")
    ax.plot(epochs, train_loss, "-",  label="Train")
    ax.plot(epochs, val_loss,   "--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Train vs. Validation Loss")
    ax.legend()

    fig.tight_layout()
    savefig(fig, os.path.join(output_dir, "fig1_loss_curves.pdf"))


# ── Figure 2: FPR and Precision vs epoch ──────────────────────────────────

def fig_fpr_precision_curves(history: list, output_dir: str):
    epochs = [ep["epoch"] for ep in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: FPR per scheduler
    ax = axes[0]
    for name, color, label in zip(SCHEDULER_NAMES, SCHED_COLORS, SCHED_LABELS):
        vals = extract_history_series(history, f"fpr/{name}", "val")
        ax.plot(epochs, vals, color=color, label=label)
    avg = extract_history_series(history, "fpr/avg", "val")
    ax.plot(epochs, avg, "k--", linewidth=2.2, label="Avg")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("Validation FPR per Scheduler")
    ax.legend()

    # Right: Precision per scheduler
    ax = axes[1]
    for name, color, label in zip(SCHEDULER_NAMES, SCHED_COLORS, SCHED_LABELS):
        vals = extract_history_series(history, f"precision/{name}", "val")
        ax.plot(epochs, vals, color=color, label=label)
    avg = extract_history_series(history, "precision/avg", "val")
    ax.plot(epochs, avg, "k--", linewidth=2.2, label="Avg")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.set_title("Validation Precision per Scheduler")
    ax.set_ylim([0, 1.05])
    ax.legend()

    fig.tight_layout()
    savefig(fig, os.path.join(output_dir, "fig2_fpr_precision_curves.pdf"))


# ── Figure 3: Precision–Recall by task-set size n ─────────────────────────

def fig_precision_recall_by_n(gen_results: dict, output_dir: str):
    """
    Replicates the key generalization figure from SAFE-TSFormer (their Fig. 2)
    but for IMC-Former: precision and recall vs n for each scheduler.
    """
    # Collect data from all gen splits that have per_n breakdowns
    n_vals, prec_vals, rec_vals = [], [], []

    for split_name, split_data in gen_results.items():
        if split_name == "test_in_dist":
            continue
        if "per_n" not in split_data:
            continue
        for n_str, metrics in split_data["per_n"].items():
            n = int(n_str)
            n_vals.append(n)
            prec_vals.append(metrics.get("precision/avg", float("nan")))
            rec_vals.append(metrics.get("recall/avg", float("nan")))

    if not n_vals:
        print("  [WARN] No per_n generalization data found — skipping fig3")
        return

    order = np.argsort(n_vals)
    n_sorted    = [n_vals[i] for i in order]
    prec_sorted = [prec_vals[i] for i in order]
    rec_sorted  = [rec_vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(n_sorted, prec_sorted, "s-", color="#d62728", label="Precision", markersize=6)
    ax.plot(n_sorted, rec_sorted,  "o-", color="#1f77b4", label="Recall",    markersize=6)
    ax.axvline(x=20.5, color="gray", linestyle=":", linewidth=1.5,
               label="Training boundary (n=20)")
    ax.set_xlabel("Task-Set Size (n)")
    ax.set_ylabel("Metric Value")
    ax.set_title("Precision and Recall vs. Task-Set Size (Generalization)")
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(8))

    fig.tight_layout()
    savefig(fig, os.path.join(output_dir, "fig3_precision_recall_by_n.pdf"))


# ── Figure 4: FPR by n across schedulers ──────────────────────────────────

def fig_fpr_by_n(gen_results: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for sname, color, label in zip(SCHEDULER_NAMES, SCHED_COLORS, SCHED_LABELS):
        n_vals, fpr_vals = [], []
        for split_name, split_data in gen_results.items():
            if split_name == "test_in_dist":
                continue
            if "per_n" not in split_data:
                continue
            for n_str, metrics in split_data["per_n"].items():
                n = int(n_str)
                fpr = metrics.get(f"fpr/{sname}", float("nan"))
                n_vals.append(n)
                fpr_vals.append(fpr)

        if not n_vals:
            continue
        order = np.argsort(n_vals)
        ax.plot([n_vals[i] for i in order],
                [fpr_vals[i] for i in order],
                "o-", color=color, label=label, markersize=5)

    ax.axvline(x=20.5, color="gray", linestyle=":", linewidth=1.5,
               label="Training boundary (n=20)")
    ax.set_xlabel("Task-Set Size (n)")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("FPR vs. Task-Set Size per Scheduling Policy")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(8))
    fig.tight_layout()
    savefig(fig, os.path.join(output_dir, "fig4_fpr_by_n.pdf"))


# ── Figure 5: Per-scheduler bar chart (in-dist test) ──────────────────────

def fig_per_scheduler_bars(gen_results: dict, output_dir: str):
    test = gen_results.get("test_in_dist", {})
    if not test:
        print("  [WARN] No test_in_dist data — skipping fig5")
        return

    metrics = ["fpr", "precision", "recall", "f1", "accuracy"]
    labels  = ["FPR", "Precision", "Recall", "F1", "Accuracy"]
    x       = np.arange(len(SCHED_LABELS))
    width   = 0.15

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (met, lbl) in enumerate(zip(metrics, labels)):
        vals = [test.get(f"{met}/{sn}", 0.0) for sn in SCHEDULER_NAMES]
        offset = (i - len(metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=lbl, alpha=0.85)
        # Annotate bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(SCHED_LABELS)
    ax.set_ylabel("Metric Value")
    ax.set_title("Per-Scheduler Metrics (In-Distribution Test Set)")
    ax.set_ylim([0, 1.18])
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    savefig(fig, os.path.join(output_dir, "fig5_per_scheduler_bars.pdf"))


# ── Figure 6: Recall vs FPR (safety tradeoff) per scheduler ───────────────

def fig_safety_tradeoff(gen_results: dict, output_dir: str):
    """
    Scatter: one point per (scheduler, split-size group) showing FPR vs Recall.
    Reveals the safety-conservatism tradeoff as n grows.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for sname, color, label in zip(SCHEDULER_NAMES, SCHED_COLORS, SCHED_LABELS):
        fprs, recs, ns = [], [], []
        # In-dist test point
        test = gen_results.get("test_in_dist", {})
        if test:
            fprs.append(test.get(f"fpr/{sname}", float("nan")))
            recs.append(test.get(f"recall/{sname}", float("nan")))
            ns.append(10)   # representative n for in-dist

        for split_name, split_data in gen_results.items():
            if split_name == "test_in_dist":
                continue
            overall = split_data.get("overall", split_data)
            fprs.append(overall.get(f"fpr/{sname}", float("nan")))
            recs.append(overall.get(f"recall/{sname}", float("nan")))

        ax.scatter(fprs, recs, color=color, label=label,
                   s=60, alpha=0.8, zorder=3)
        if len(fprs) > 1:
            ax.plot(fprs, recs, color=color, alpha=0.4, linewidth=1)

    ax.set_xlabel("False Positive Rate (FPR)  →  lower is safer")
    ax.set_ylabel("Recall  →  higher is better coverage")
    ax.set_title("Safety–Coverage Tradeoff Across Task-Set Sizes")
    ax.set_xlim([-0.01, None])
    ax.set_ylim([0, 1.05])
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.5,
               label="FPR = 0 (ideal)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, os.path.join(output_dir, "fig6_safety_tradeoff.pdf"))


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history_json", required=True,
                   help="Path to *_history.json written by trainer")
    p.add_argument("--gen_json",     required=True,
                   help="Path to *_gen_results.json written by trainer")
    p.add_argument("--output_dir",   default="figures")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading history from {args.history_json}")
    history = load_json(args.history_json)
    print(f"Loading gen results from {args.gen_json}")
    gen_results = load_json(args.gen_json)

    print("\nGenerating figures...")
    fig_loss_curves(history, args.output_dir)
    fig_fpr_precision_curves(history, args.output_dir)
    fig_precision_recall_by_n(gen_results, args.output_dir)
    fig_fpr_by_n(gen_results, args.output_dir)
    fig_per_scheduler_bars(gen_results, args.output_dir)
    fig_safety_tradeoff(gen_results, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
