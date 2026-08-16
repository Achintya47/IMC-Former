# IMC-Former: Execution Guide

Criticality-Aware Dual-Stream Transformer for Imprecise Mixed-Criticality
Schedulability Prediction.

---

## 1. Project Layout

```
imc_former/
│
├── configs/
│   └── config.py            # All hyperparameters (ModelConfig, LossConfig, TrainingConfig)
│
├── data/
│   └── dataset.py           # IMCDataset, DataLoader builder, curriculum sampler
│
├── models/
│   ├── imc_former.py        # Full model: encoder, dual-stream, cross-attn, heads
│   └── loss.py              # Safety-aware IMC loss
│
├── training/
│   └── trainer.py           # Training loop, logging, checkpointing
│
├── evaluation/
│   └── metrics.py           # FPR, Precision, Recall, F1, Accuracy, AUROC
│
├── prepare_data.py          # Step 1 — merge your CSVs into train/val/test/gen splits
├── train.py                 # Step 2 — train and evaluate the model
├── visualize.py             # Step 3 — generate paper figures
│
├── requirements.txt
├── DATASET_SPEC.md          # Full dataset specification and generation protocol
└── README.md                # This file
```

Runtime output directories (created automatically):
```
data/           ← split CSVs produced by prepare_data.py
logs/           ← training log files + JSON history + TensorBoard events
checkpoints/    ← saved model weights (.pt files)
figures/        ← PDF figures produced by visualize.py
```

---

## 2. Installation

```bash
pip install -r requirements.txt
```

Minimum Python: 3.9.  Tested with PyTorch 2.0–2.13, CUDA 11.8+.

For CUDA training (strongly recommended for the full model):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## 3. Your CSV Files

### Naming convention

Your files must be named:   `<prefix>_<n>.csv`

Examples that all work:
```
imc_4.csv        tasksets_4.csv       data_4.csv
imc_8.csv        tasksets_8.csv       data_8.csv
imc_12.csv       tasksets_12.csv      data_12.csv
```

The script searches for `*_<n>.csv` inside the directory you point it at.
Multiple files for the same n are concatenated automatically.

Place all raw files in one flat directory, e.g.:
```
raw_data/
  imc_4.csv
  imc_8.csv
  imc_12.csv
  imc_16.csv
  imc_20.csv
  imc_24.csv      ← generalization sizes (not used in training)
  imc_32.csv
  imc_64.csv
```

### Required columns

**Set-level** (one value per row):
```
Taskset_ID        int    unique row id
Num_Tasks         int    number of tasks n
U_total           float  total LO utilisation Σ(CLO_i/T_i)
HI_ratio          float  fraction of tasks with Crit_i = 1
CF                float  global criticality inflation factor
```

**Per-task** (repeated for i = 1 … n):
```
CLO{i}    int     mandatory execution time (LO WCET)
CHI{i}    int     full execution time (HI WCET); 0 for pure LO tasks
D{i}      int     relative deadline
T{i}      int     period
Crit{i}   int     criticality level: 0 = LO, 1 = HI
```

**Labels** (four binary targets):
```
AMC_IMC_Schedulable      {0, 1}
TT_Merge_Schedulable     {0, 1}
IMC_PnG_Schedulable      {0, 1}
EDF_IMC_Schedulable      {0, 1}
```

The column order does not matter. Extra columns are ignored.

### Handling different n values in the same file

If you have a single file containing mixed n values (e.g., rows with n=4 and n=8),
split it into separate `_4.csv` and `_8.csv` files first, or use the `Num_Tasks`
column — the code reads `Num_Tasks` per row and pads accordingly.

---

## 4. Step 1 — Prepare Data Splits

```bash
python prepare_data.py \
    --input_dir  raw_data/ \
    --output_dir data/ \
    --train_ns   4 8 12 16 20 \
    --gen_small  24 28 \
    --gen_medium 32 40 \
    --gen_large  48 64 \
    --gen_xlarge 128 256 \
    --gen_xxlarge 512 1000 \
    --train_frac 0.70 \
    --val_frac   0.15 \
    --seed       42
```

### What it produces in `data/`

| File                | Content                                 |
|---------------------|-----------------------------------------|
| `train.csv`         | 70% of in-distribution rows             |
| `val_in_dist.csv`   | 15% of in-distribution rows             |
| `test_in_dist.csv`  | 15% of in-distribution rows             |
| `gen_small.csv`     | All rows from `--gen_small` sizes       |
| `gen_medium.csv`    | All rows from `--gen_medium` sizes      |
| `gen_large.csv`     | All rows from `--gen_large` sizes       |
| `gen_xlarge.csv`    | All rows from `--gen_xlarge` sizes      |
| `gen_xxlarge.csv`   | All rows from `--gen_xxlarge` sizes     |
| `data_stats.json`   | Row counts + class balance per split    |

**Generalization files are never touched during training** — they are evaluated
only after the final best checkpoint is loaded.

### Parameter reference

| Parameter       | Default      | Meaning |
|-----------------|--------------|---------|
| `--input_dir`   | *required*   | Directory containing your `*_n.csv` files |
| `--output_dir`  | `data/`      | Where split CSVs are written |
| `--train_ns`    | 4 8 12 16 20 | Task-set sizes used for training |
| `--gen_small`   | 24 28        | Near-OOD generalization sizes |
| `--gen_medium`  | 32 40        | Medium OOD sizes |
| `--gen_large`   | 48 64        | Large OOD sizes |
| `--gen_xlarge`  | 128 256      | Extreme extrapolation sizes |
| `--gen_xxlarge` | 512 1000     | Upper-bound extrapolation |
| `--train_frac`  | 0.70         | Training fraction of in-dist data |
| `--val_frac`    | 0.15         | Validation fraction (test = 1 - train - val) |
| `--seed`        | 42           | Random seed for reproducible splits |

Missing generalization sizes (no matching file found) are skipped silently.

---

## 5. Step 2 — Train the Model

```bash
python train.py
```

This uses all defaults. The most common overrides:

```bash
# Custom data and experiment name
python train.py \
    --data_dir  data/ \
    --run_name  imc_former_exp1 \
    --num_epochs 80

# GPU training (default; uses first available CUDA device)
python train.py --device cuda

# CPU training (e.g., debugging)
python train.py --device cpu --num_workers 0

# Increase batch size (if GPU has enough VRAM)
python train.py --batch_size 128

# Tighten FP penalty (drives FPR lower, reduces recall)
python train.py --fp_lambda 5.0 --hi_mu 6.0

# Evaluation only — load existing best checkpoint, skip training
python train.py --eval_only --run_name imc_former_exp1
```

### What gets written

```
logs/
  imc_former_base.log                  Full console + debug log (every batch)
  imc_former_base_history.json         Per-epoch train/val metrics
  imc_former_base_gen_results.json     Generalization split metrics + per-n breakdown
  tensorboard/imc_former_base/         TensorBoard event files

checkpoints/
  imc_former_base_best.pt              Best model (lowest val FPR, acc ≥ threshold)
  imc_former_base_epoch10.pt           Periodic checkpoints every 10 epochs
  imc_former_base_epoch20.pt
  ...
```

### Parameter reference

| Parameter       | Default            | Meaning |
|-----------------|--------------------|---------|
| `--data_dir`    | `data/`            | Directory with split CSVs from prepare_data.py |
| `--log_dir`     | `logs/`            | Where logs and JSON results are written |
| `--ckpt_dir`    | `checkpoints/`     | Where model checkpoints are saved |
| `--fig_dir`     | `figures/`         | Reserved for figure output |
| `--run_name`    | `imc_former_base`  | Prefix for all output filenames |
| `--batch_size`  | `64`               | Training batch size |
| `--num_epochs`  | `80`               | Maximum training epochs |
| `--encoder_lr`  | `3e-4`             | Learning rate for encoder + transformer layers |
| `--head_lr`     | `1e-3`             | Learning rate for scheduler-specific prediction heads |
| `--weight_decay`| `1e-4`             | AdamW weight decay |
| `--seed`        | `42`               | Random seed |
| `--device`      | `cuda`             | `cuda` or `cpu` |
| `--num_workers` | `4`                | DataLoader worker processes (use 0 on Windows) |
| `--fp_lambda`   | `2.0`              | FP penalty weight λ in the loss |
| `--hi_mu`       | `3.0`              | Additional HI-critical FP penalty weight μ |
| `--hi_thresh`   | `0.7`              | U_HI threshold above which a task set is "HI-critical" |
| `--d_model`     | `256`              | Embedding dimension throughout the model |
| `--n_heads`     | `8`                | Number of attention heads |
| `--n_layers`    | `3`                | Transformer blocks per stream |
| `--chunk_size`  | `8`                | Tasks per chunk in hierarchical pooling |
| `--eval_only`   | `False`            | Skip training; load best checkpoint and evaluate |

### TensorBoard monitoring

While training is running (or after), open a second terminal:
```bash
tensorboard --logdir logs/tensorboard/
```
Then open http://localhost:6006 in your browser.

Key scalars to watch:
- `val/fpr/avg`       — primary metric (lower = better, this is what saves best model)
- `val/precision/avg` — safety proxy (keep this near 1.0)
- `val/recall/avg`    — coverage (decreases as FPR is pushed down)
- `train/loss/total`  vs `val/loss/total` — check for overfitting
- `lr/encoder`, `lr/heads` — learning rate decay events

---

## 6. Step 3 — Generate Figures

```bash
python visualize.py \
    --history_json logs/imc_former_base_history.json \
    --gen_json     logs/imc_former_base_gen_results.json \
    --output_dir   figures/
```

Produces (all as 300 DPI PDFs, suitable for paper submission):

| File | Content |
|------|---------|
| `fig1_loss_curves.pdf`            | Training loss components over epochs |
| `fig2_fpr_precision_curves.pdf`   | FPR and Precision vs epoch per scheduler |
| `fig3_precision_recall_by_n.pdf`  | Precision/Recall vs task-set size (generalization) |
| `fig4_fpr_by_n.pdf`               | FPR vs task-set size per scheduling policy |
| `fig5_per_scheduler_bars.pdf`     | Per-scheduler metric bar chart (in-dist test) |
| `fig6_safety_tradeoff.pdf`        | FPR vs Recall scatter across n sizes |

---

## 7. Things to Keep in Mind

### Dataset generation
- **Class balance matters.** Target 40–60% schedulable per label.  
  The `data_stats.json` written by `prepare_data.py` shows the balance per split.  
  If any scheduler label is >85% or <15% schedulable, your dataset is imbalanced —
  the BCE positive class weights (printed at training start) will compensate, but
  generating a more balanced dataset is strongly preferred.

- **Use constrained deadlines** (D_i ≤ T_i) for the majority of task sets.  
  Implicit-deadline (D_i = T_i) tasks make EDF schedulability trivial (just check
  Σ U_i ≤ 1), which the model learns trivially. The hard and interesting cases are
  constrained-deadline, where no closed-form test exists.

- **U_total range.** Span [0.1, 1.4] so the dataset contains clearly schedulable,
  boundary, and clearly unschedulable examples. See `DATASET_SPEC.md` for the
  full generation protocol.

### Training
- **num_workers.** On Windows, set `--num_workers 0` (multiprocessing spawn issues).  
  On Linux/macOS, 4–8 workers are fine.

- **VRAM.** The full model (d_model=256, 3 transformer layers) requires approximately
  4 GB VRAM at batch_size=64 with n_max=20. For n_max=64 use batch_size=32.

- **Early stopping watches FPR**, not loss. If your dataset has many easy examples
  (low-utilisation task sets that are trivially schedulable), FPR may bottom out early
  and training will stop. This is correct behaviour — the model has learned what it can.

- **Best model criterion.** The checkpoint saved as `_best.pt` is the one with the
  lowest `val/fpr/avg` subject to `val/accuracy/avg >= 0.80`. If accuracy never
  reaches 0.80, no best checkpoint is saved (only periodic epoch checkpoints).
  Lower the threshold in `configs/config.py → TrainingConfig.min_accuracy_threshold`
  if this happens.

- **Curriculum phases.** The sampler automatically adjusts:
  - Epochs 1–20: over-samples n=4,8 (simple cases, fast convergence)
  - Epochs 21–60: uniform across all n
  - Epochs 61+: over-samples n=16,20 (boundary of training distribution)
  These boundaries are controlled by `phase1_end_epoch` and `phase2_end_epoch`
  in `TrainingConfig`.

### Loss hyperparameters
- `--fp_lambda` (λ): The core FP penalty. Higher → lower FPR but lower recall.  
  Start at 2.0. If FPR is still above 5% after training, try 5.0. If recall drops
  below 60%, reduce to 1.0.

- `--hi_mu` (μ): Extra penalty for HI-critical task sets. Always set μ ≥ λ.  
  A value of 3× λ is a reasonable starting point.

- `--hi_thresh`: A task set is "HI-critical" if any HI task has normalised U_HI_i
  above this threshold. After normalisation U_HI_i is not in the original [0,1] range,
  so this threshold is approximate. The default 0.7 works well empirically.

### Model selection
The `_best.pt` checkpoint is selected on validation FPR. When you do the final
evaluation with `--eval_only`, it loads this checkpoint and runs all generalization
splits. The generalization results are the numbers you report in your paper.

### Generalization
- The model is trained on n ∈ {4, 8, 12, 16, 20}. Anything larger is OOD.
- Precision is expected to remain near 1.0 (no unsafe predictions) even as n grows,
  while recall decreases (more conservative). This is the intended behaviour.
- If precision drops below 0.90 on OOD sizes, the model is making unsafe predictions —
  increase `--fp_lambda`.

---

## 8. Quick-Start Checklist

```
[ ] Place CSVs in   raw_data/  named  XYZ_4.csv, XYZ_8.csv, ...
[ ] pip install -r requirements.txt
[ ] python prepare_data.py --input_dir raw_data/ --output_dir data/
[ ] python train.py --run_name my_experiment --device cuda
[ ] Monitor:  tensorboard --logdir logs/tensorboard/
[ ] After training completes:
    python visualize.py \
        --history_json logs/my_experiment_history.json \
        --gen_json     logs/my_experiment_gen_results.json
[ ] Report results from:  logs/my_experiment_gen_results.json
```

---

## 9. Ablation Experiments

After your base model is trained, ablations are run by changing one thing and
re-training with the same seed and data. Recommended ablations (from DATASET_SPEC.md):

```bash
# A1: No cross-stream attention → set n_transformer_layers=1, remove cross-attn blocks
#     (requires a small code change in models/imc_former.py — set cross streams to identity)

# A4: No FP penalty (pure BCE)
python train.py --fp_lambda 0.0 --hi_mu 0.0 --run_name ablation_no_fp_penalty

# A5: No HI-asymmetric weighting
python train.py --hi_mu 0.0 --run_name ablation_no_hi_penalty

# A6: MLP baseline (no transformer) — run with n_layers=0 and chunk_size=99999
python train.py --n_layers 0 --chunk_size 9999 --run_name ablation_mlp_only
```

Compare the resulting `gen_results.json` files to quantify each component's contribution.
