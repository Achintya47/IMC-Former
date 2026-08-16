"""
prepare_data.py
===============
Merges per-n CSV files (e.g., XYZ_4.csv, XYZ_8.csv, ...) into the
train / val / test / generalization splits expected by IMC-Former.

Usage:
    python prepare_data.py \\
        --input_dir  /path/to/raw_csvs \\
        --output_dir data/ \\
        --train_ns   4 8 12 16 20 \\
        --gen_small  24 28 \\
        --gen_medium 32 40 \\
        --gen_large  48 64 \\
        --gen_xlarge 128 256 \\
        --gen_xxlarge 512 1000 \\
        --train_frac 0.70 \\
        --val_frac   0.15 \\
        --seed       42

Input convention:
    Files must be named  <anything>_<n>.csv  e.g.  imc_4.csv, tasksets_8.csv
    The suffix _<n>.csv is parsed to identify the task-set size.
    All files in --input_dir matching *_<N>.csv for the requested N values
    are loaded and concatenated.

Output files written to --output_dir:
    train.csv             in-dist training split (n ∈ train_ns, 70%)
    val_in_dist.csv       in-dist validation   (n ∈ train_ns, 15%)
    test_in_dist.csv      in-dist test         (n ∈ train_ns, 15%)
    gen_small.csv         generalization split  (n ∈ gen_small)
    gen_medium.csv        generalization split  (n ∈ gen_medium)
    gen_large.csv         generalization split  (n ∈ gen_large)
    gen_xlarge.csv        generalization split  (n ∈ gen_xlarge)
    gen_xxlarge.csv       generalization split  (n ∈ gen_xxlarge)
    data_stats.json       row counts and class balance per split

Notes:
    - Generalization files are NOT split further; the entire file is used as-is.
    - Shuffling uses --seed for reproducibility.
    - Missing generalization sizes are skipped silently (no file written).
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd


LABEL_COLS = [
    "AMC_IMC_Schedulable",
    "TT_Merge_Schedulable",
    "IMC_PnG_Schedulable",
    "EDF_IMC_Schedulable",
]


def find_csv_for_n(input_dir: str, n: int):
    """
    Find all CSVs in input_dir whose filename ends with _{n}.csv (case-insensitive).
    Returns list of paths.
    """
    pattern = os.path.join(input_dir, f"*_{n}.csv")
    return sorted(glob.glob(pattern))


def load_and_tag(paths, n: int) -> pd.DataFrame:
    """Load CSVs, concatenate, ensure Num_Tasks column is set."""
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        if "Num_Tasks" not in df.columns:
            df["Num_Tasks"] = n
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    return out


def check_required_columns(df: pd.DataFrame, n: int):
    """Validate that all task columns and label columns are present."""
    for i in range(1, n + 1):
        for col in [f"CLO{i}", f"CHI{i}", f"D{i}", f"T{i}", f"Crit{i}"]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' missing for n={n}. "
                    f"Check your CSV format. Available cols: {list(df.columns)[:20]}"
                )
    missing_labels = [c for c in LABEL_COLS if c not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing label columns: {missing_labels}")


def class_balance_summary(df: pd.DataFrame) -> dict:
    """Compute fraction of schedulable examples per scheduler."""
    out = {}
    for col in LABEL_COLS:
        if col in df.columns:
            out[col] = float(df[col].mean())
    return out


def split_in_dist(df: pd.DataFrame, train_frac: float, val_frac: float,
                  seed: int) -> tuple:
    """Stratified random split into train / val / test."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n = len(df)
    t = int(n * train_frac)
    v = int(n * (train_frac + val_frac))
    train = df.iloc[idx[:t]].reset_index(drop=True)
    val   = df.iloc[idx[t:v]].reset_index(drop=True)
    test  = df.iloc[idx[v:]].reset_index(drop=True)
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Prepare IMC-Former data splits")
    parser.add_argument("--input_dir",   required=True,
                        help="Directory containing raw per-n CSV files")
    parser.add_argument("--output_dir",  default="data",
                        help="Output directory for split CSVs (default: data/)")
    parser.add_argument("--train_ns",    nargs="+", type=int,
                        default=[4, 8, 12, 16, 20],
                        help="Task-set sizes for in-distribution training")
    parser.add_argument("--gen_small",   nargs="+", type=int, default=[24, 28])
    parser.add_argument("--gen_medium",  nargs="+", type=int, default=[32, 40])
    parser.add_argument("--gen_large",   nargs="+", type=int, default=[48, 64])
    parser.add_argument("--gen_xlarge",  nargs="+", type=int, default=[128, 256])
    parser.add_argument("--gen_xxlarge", nargs="+", type=int, default=[512, 1000])
    parser.add_argument("--train_frac",  type=float, default=0.70)
    parser.add_argument("--val_frac",    type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stats = {}

    # ── In-distribution: load, concat, validate, split ──────────────────────
    print(f"\n{'='*60}")
    print(f"Loading in-distribution data (n ∈ {args.train_ns})")
    print(f"{'='*60}")

    in_dist_dfs = []
    for n in args.train_ns:
        paths = find_csv_for_n(args.input_dir, n)
        if not paths:
            print(f"  [WARN] No CSV found for n={n} in {args.input_dir} — skipping")
            continue
        print(f"  n={n:4d}: found {len(paths)} file(s): {[os.path.basename(p) for p in paths]}")
        df = load_and_tag(paths, n)
        try:
            check_required_columns(df, n)
        except ValueError as e:
            print(f"  [ERROR] {e}")
            sys.exit(1)
        in_dist_dfs.append(df)
        print(f"          Loaded {len(df):,} rows | Balance: {class_balance_summary(df)}")

    if not in_dist_dfs:
        print("[ERROR] No in-distribution data loaded. Exiting.")
        sys.exit(1)

    in_dist = pd.concat(in_dist_dfs, ignore_index=True)
    # Shuffle before split
    in_dist = in_dist.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    train, val, test = split_in_dist(
        in_dist, args.train_frac, args.val_frac, args.seed
    )

    for split_name, df_split in [("train", train), ("val_in_dist", val),
                                  ("test_in_dist", test)]:
        path = os.path.join(args.output_dir, f"{split_name}.csv")
        df_split.to_csv(path, index=False)
        bal  = class_balance_summary(df_split)
        stats[split_name] = {"n_rows": len(df_split), "class_balance": bal}
        print(f"\n  {split_name:20s}: {len(df_split):>8,} rows → {path}")
        print(f"  {'Balance':20s}: {bal}")

    # ── Generalization splits ─────────────────────────────────────────────────
    gen_groups = {
        "gen_small":   args.gen_small,
        "gen_medium":  args.gen_medium,
        "gen_large":   args.gen_large,
        "gen_xlarge":  args.gen_xlarge,
        "gen_xxlarge": args.gen_xxlarge,
    }

    print(f"\n{'='*60}")
    print("Loading generalization splits")
    print(f"{'='*60}")

    for gname, ns in gen_groups.items():
        gen_dfs = []
        for n in ns:
            paths = find_csv_for_n(args.input_dir, n)
            if not paths:
                print(f"  [INFO] No file for n={n} — skipping")
                continue
            print(f"  {gname} n={n}: {[os.path.basename(p) for p in paths]}")
            df = load_and_tag(paths, n)
            gen_dfs.append(df)

        if not gen_dfs:
            print(f"  [WARN] {gname}: no data found, file not written")
            continue

        gen_df = pd.concat(gen_dfs, ignore_index=True)
        gen_df = gen_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        path   = os.path.join(args.output_dir, f"{gname}.csv")
        gen_df.to_csv(path, index=False)
        bal    = class_balance_summary(gen_df)
        stats[gname] = {"n_rows": len(gen_df), "class_balance": bal}
        print(f"  {gname:20s}: {len(gen_df):>8,} rows → {path}")
        print(f"  {'Balance':20s}: {bal}")

    # ── Summary ───────────────────────────────────────────────────────────────
    stats_path = os.path.join(args.output_dir, "data_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    total = sum(v["n_rows"] for v in stats.values())
    for k, v in stats.items():
        print(f"  {k:25s}: {v['n_rows']:>8,} rows")
    print(f"  {'TOTAL':25s}: {total:>8,} rows")
    print(f"\nStats written to {stats_path}")
    print("\nDone. You can now run:  python train.py")


if __name__ == "__main__":
    main()
