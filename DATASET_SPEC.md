# IMC-Former: Dataset Specification and Training Strategy

## 1. Dataset Column Schema

Each row in the CSV represents one IMC task set. The schema is:

```
Taskset_ID    : int    — unique identifier
Num_Tasks     : int    — number of tasks n ∈ {4,8,12,16,20,24,28,32,...}
U_total       : float  — total LO-mode utilization Σ(CLO_i/T_i)
HI_ratio      : float  — fraction of tasks with Crit_i = 1
CF            : float  — global criticality inflation factor (CHI/CLO average)

Per task i ∈ {1,...,n}:
  CLO_i  : int    — mandatory execution time (LO WCET)
  CHI_i  : int    — full execution time (HI WCET; 0 if pure LO task)
  D_i    : int    — relative deadline
  T_i    : int    — period
  Crit_i : int    — criticality level (0=LO, 1=HI)

Labels (one per IMC scheduling algorithm):
  AMC_IMC_Schedulable  : {0,1}
  TT_Merge_Schedulable : {0,1}
  IMC_PnG_Schedulable  : {0,1}
  EDF_IMC_Schedulable  : {0,1}
```

## 2. Dataset Splits Required

### Training Corpus (generate these, mix all n values)

| Split Name        | Task-set sizes n         | Count      | Purpose                        |
|-------------------|--------------------------|------------|--------------------------------|
| train             | {4, 8, 12, 16, 20}       | 250,000    | Model parameter learning       |
| val_in_dist       | {4, 8, 12, 16, 20}       | 37,500     | Val loss, early stopping, LR   |
| test_in_dist      | {4, 8, 12, 16, 20}       | 37,500     | In-distribution metrics        |

Total in-distribution corpus: **325,000 task sets**  
Split ratio: 70% train / 15% val / 15% test  
All three splits drawn from the same n ∈ {4,8,12,16,20} distribution.

### Generalization Corpus (strictly held out — never seen during training)

| Split Name        | Task-set sizes n         | Count      | Purpose                            |
|-------------------|--------------------------|------------|------------------------------------|
| gen_small         | {24, 28}                 | 30,000     | Near-OOD generalization            |
| gen_medium        | {32, 40}                 | 30,000     | Medium OOD generalization          |
| gen_large         | {48, 64}                 | 25,000     | Large OOD generalization           |
| gen_xlarge        | {128, 256}               | 15,000     | Extreme extrapolation              |
| gen_xxlarge       | {512, 1000}              | 5,000      | Upper bound test                   |

Total generalization corpus: **105,000 task sets**  
These are **never touched** until after training is fully complete.

### Per-n counts (within training corpus, balanced)
- n=4:   50,000 task sets
- n=8:   50,000 task sets
- n=12:  50,000 task sets
- n=16:  50,000 task sets
- n=20:  50,000 task sets

### Class balance target (per label, per split)
- Target: 40–60% schedulable per label after filtering
- U_total should span [0.1, 1.4] uniformly to ensure both regions
- HI_ratio should span [0.1, 0.9] uniformly

## 3. Generation Protocol (for the dataset generator)

```
For each task set of size n:
  1. Sample U_total ~ Uniform(0.1, 1.4)
  2. Sample HI_ratio ~ Uniform(0.1, 0.9); n_HI = round(HI_ratio * n)
  3. Sample {U_i^LO}_{i=1}^n via UUniFast(n, U_total)
  4. For each task i:
       T_i ~ LogUniform(1, 1000)
       CLO_i = floor(U_i^LO * T_i), ensure CLO_i >= 1
       if task i is HI (i.e., i <= n_HI after random permutation):
           CF_i ~ Uniform(1.2, 4.0)
           CHI_i = ceil(CLO_i * CF_i)
           Crit_i = 1
       else:
           CHI_i = 0
           Crit_i = 0
       D_i ~ Uniform(CLO_i, T_i)  # constrained deadline
  5. Compute ground-truth labels via exact schedulability tests
  6. Discard task sets where U_total^LO > 1.5 (infeasible by inspection)
```

## 4. Training Strategy

### 4.1 Curriculum Learning: Mixed-n Training

We use a **mixed-n curriculum** rather than training on each n separately.
All task-set sizes {4,8,12,16,20} are trained simultaneously in each batch.
Within each batch, we sample task sets proportionally:
  - Phase 1 (epochs 1–20):  over-sample small n (4, 8) at 40% each, larger n at 7% each
  - Phase 2 (epochs 21–60): uniform sampling across all n
  - Phase 3 (fine-tune):    over-sample n=16 and n=20 (boundary of training distribution)

**Rationale**: curriculum from small to large n helps the model first learn the local 
schedulability structure (short deadline interactions, single-task dominance) before 
learning global set-level properties. This mirrors how analytical tests work: 
small-n exact tests are easy to learn; large-n requires the hierarchical interference 
reasoning the transformer must generalize.

### 4.2 Variable-Length Batching

Within a batch, all task sets are padded to the maximum n in that batch.
A binary mask M ∈ {0,1}^N is passed alongside embeddings, where M_i = 1 for 
real tasks and M_i = 0 for padding. Padding tasks contribute zero to attention 
and pooling.

Batch composition strategy: each batch contains task sets of AT MOST 2 different n values.
This prevents high-n task sets from dominating gradient computation.

### 4.3 Optimizer and Learning Rate

- Optimizer: AdamW (weight_decay=1e-4)
- Initial LR: 3e-4 for encoder, 1e-3 for prediction heads
- Schedule: ReduceLROnPlateau on validation FPR (not loss — we care about FPR)
- Early stopping: patience=10 on validation FPR
- Gradient clipping: max_norm=1.0

### 4.4 Loss Weighting Strategy

The safety-aware loss is:
  L = Σ_S w_S · BCE(y_S, ŷ_S) + λ · (1/B) Σ_b Σ_S (1-y_{b,S}) · ŷ_{b,S}
    + μ · FP_HI_penalty

Where:
  w_S = 1.0 for all schedulers (equal scheduler weighting; adjust after first run)
  λ = 2.0 (false-positive penalty strength; tune in {1.0, 2.0, 5.0, 10.0})
  μ = 3.0 (HI-mode critical false-positive penalty multiplier)

### 4.5 Model Selection Criterion

Best model = checkpoint with lowest FPR on val_in_dist, 
             subject to: accuracy >= 85% on val_in_dist.
We do NOT select on BCE loss — this would optimize for accuracy, not safety.

## 5. Evaluation Metrics (in order of importance)

1. FPR (False Positive Rate) = FP / (FP + TN)  — PRIMARY METRIC
2. Precision = TP / (TP + FP)                   — safety proxy
3. Recall = TP / (TP + FN)                      — coverage
4. F1 Score                                      — balance
5. Accuracy                                      — overall correctness
6. AUROC                                         — threshold-independent

Reported separately per scheduler head AND aggregated.
Reported separately for:
  - In-distribution (test_in_dist)
  - Each generalization split (gen_small, gen_medium, gen_large, ...)

## 6. Ablation Experiments (post base-model training)

After base training, run ablations to isolate contributions:
  A1: Remove cross-stream attention (replace with flat transformer)
  A2: Remove separate stream pooling (use single attention pool)
  A3: Remove set-level context features (z_set)
  A4: Remove FP penalty (train with BCE only)
  A5: Remove HI-asymmetric weighting
  A6: MLP-only encoder (no transformer at all) — baseline

Each ablation retrains from scratch with identical hyperparameters.
