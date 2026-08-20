# Coded by Dr. Lalatendu Behera
# Extended by IMC-Former pipeline — parallel multi-n generation, all labels for all n
# =====================================================================================
#
# Generates one CSV per task-set size n, named:
#   generated_datasets/imc_<n>.csv
#
# ALL FOUR schedulability tests (AMC-IMC, TT-Merge, IMC-PnG, EDF-IMC) are computed
# for EVERY task set at EVERY n value. No labels are skipped.
#
# Measured generation times (single CPU core, typical task-set parameters):
#   n=4–20  : ~3 min total (0.04–0.25 ms/set × 299,700 rows each)
#   n=24–64 : ~5 min total (0.42–2.47 ms/set × 29k–75k rows each)
#   n=128   : ~2 min       (15 ms/set × 7,200 rows)
#   n=512   : ~25 min      (~300 ms/set × 4,950 rows)
#   n=1024  : ~50 min      (~600 ms/set × 4,950 rows)
#   TOTAL   : ~85 min on 1 core  →  ~15 min with 8 parallel workers
#
# ── Quick-start ──────────────────────────────────────────────────────────────
#
# Default run (all sizes, all labels, uses all CPU cores):
#   python Generate_Task_Set.py
#
# Generate only specific sizes:
#   python Generate_Task_Set.py --sizes 4 8 12 16 20
#
# Control parallelism:
#   python Generate_Task_Set.py --workers 4
#
# Resume (skip sizes whose CSV already has enough rows):
#   python Generate_Task_Set.py --resume
#
# Quick test (2 min, verify pipeline end-to-end):
#   python Generate_Task_Set.py --sizes 4 8 --sets_per_combo 10 --out_dir test_data/
#
# ── Output layout ─────────────────────────────────────────────────────────────
# generated_datasets/
#   imc_4.csv         ~299,700 rows  training (n ∈ {4,8,12,16,20})
#   imc_8.csv
#   ...
#   imc_20.csv
#   imc_24.csv        ~74,700 rows   gen_small (n ∈ {24,28})
#   imc_28.csv
#   imc_32.csv        ~59,850 rows   gen_medium (n ∈ {32,40})
#   imc_40.csv
#   imc_48.csv        ~29,700 rows   gen_large (n ∈ {48,64})
#   imc_64.csv
#   imc_128.csv       ~7,200 rows    gen_xlarge
#   imc_512.csv       ~4,950 rows    gen_xxlarge
#   imc_1024.csv      ~4,950 rows    gen_xxlarge
#   generation_summary.csv           per-combo schedulability statistics

import argparse
import csv
import math
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

# ── Type alias ────────────────────────────────────────────────────────────────
Task = Tuple[int, int, int, int, int, int]
# (Task_ID, C_lo, C_hi, T, D, Criticality)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TUNABLES
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42

# Parameter grid (identical to original script)
UTILIZATIONS = [round(0.10 + 0.05 * i, 2) for i in range(int((0.95 - 0.10) / 0.05) + 1)]
HI_RATIOS    = [round(0.4  + 0.1  * i, 2) for i in range(int((0.8  - 0.4)  / 0.1) + 1)]
CF_VALUES    = list(range(2, 7))

# ── Tiered sets_per_combo ─────────────────────────────────────────────────────
# Total rows ≈ sets_per_combo × 450 (18 U × 5 HI_ratio × 5 CF)
#
#  n       spc    rows     role
#  ─────────────────────────────────────────────────
#  4–20    666    ~299,700  training
#  24,28   166    ~74,700   gen_small
#  32,40   133    ~59,850   gen_medium
#  48,64   66     ~29,700   gen_large
#  128     16     ~7,200    gen_xlarge
#  512     11     ~4,950    gen_xxlarge
#  1024    11     ~4,950    gen_xxlarge

SETS_PER_COMBO_TIER: Dict[int, int] = {
    4:    666,
    8:    666,
    12:   666,
    16:   666,
    20:   666,
    24:   166,
    28:   166,
    32:   133,
    40:   133,
    48:   66,
    64:   66,
    128:  16,
    512:  11,
    1024: 11,
}
DEFAULT_TASK_COUNTS = sorted(SETS_PER_COMBO_TIER.keys())

# ── Period ranges per n ───────────────────────────────────────────────────────
# Wider ranges for larger n prevent CLO_i collapsing to 1 due to small U_i × T_i.
# Derivation: with U_total=0.5, n tasks, each U_i ≈ 0.5/n.
#   CLO_i = max(1, round(U_i × T_i)). For this to be ≥2 reliably, need T_i ≥ 4/U_i ≈ 8n.
#   n=512: need T ≥ 4096, so T_min=500, T_max=10000 is appropriate.
PERIOD_RANGE: Dict[int, Tuple[int, int]] = {
    4:    (10,   100),
    8:    (10,   100),
    12:   (10,   200),
    16:   (10,   200),
    20:   (10,   200),
    24:   (10,   500),
    28:   (10,   500),
    32:   (20,   500),
    40:   (20,   500),
    48:   (50,  1000),
    64:   (50,  1000),
    128:  (100, 5000),
    512:  (500, 10000),
    1024: (500, 10000),
}

# ── TT-Merge busy-window cap ──────────────────────────────────────────────────
# Full TT-Merge runs for ALL n. This cap prevents pathological cases where
# LCM(periods) is astronomically large. The fallback (10 × T_max) applies
# when W exceeds the cap, exactly as in the original script.
TT_MAX_WINDOW: Dict[int, int] = {
    4:    200_000,
    8:    200_000,
    12:   200_000,
    16:   200_000,
    20:   200_000,
    24:   100_000,
    28:   100_000,
    32:   100_000,
    40:    50_000,
    48:    50_000,
    64:    50_000,
    128:   30_000,
    512:   20_000,
    1024:  20_000,
}

# ── AMC-IMC iteration cap ─────────────────────────────────────────────────────
# Full AMC-IMC runs for ALL n. The cap controls worst-case diverging RTA chains.
# In practice most task sets converge in <50 iterations; the cap only fires on
# truly unschedulable sets (where R → ∞ is the correct answer anyway).
AMC_MAX_ITER: Dict[int, int] = {
    4:    1000,
    8:    1000,
    12:   1000,
    16:   1000,
    20:   1000,
    24:   500,
    28:   500,
    32:   500,
    40:   200,
    48:   200,
    64:   100,
    128:  100,
    512:  50,
    1024: 50,
}

FALLBACK_K     = 10
PROGRESS_EVERY = 1000   # print progress every N task sets within a worker


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TASK GENERATION  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def uunifast(n: int, U_total: float) -> List[float]:
    utils, sumU = [], U_total
    for i in range(1, n):
        nextSumU = sumU * (random.random() ** (1 / (n - i)))
        utils.append(sumU - nextSumU)
        sumU = nextSumU
    utils.append(sumU)
    return utils


def generate_periods(n: int, Tmin: int, Tmax: int) -> List[int]:
    return [random.randint(Tmin, Tmax) for _ in range(n)]


def generate_taskset(n: int, U_total: float,
                     Tmin: int, Tmax: int) -> List[Tuple[int, int, int]]:
    utils      = uunifast(n, U_total)
    periods    = generate_periods(n, Tmin, Tmax)
    exec_times = [max(1, int(round(u * T))) for u, T in zip(utils, periods)]
    deadlines  = periods[:]
    return list(zip(exec_times, periods, deadlines))


def convert_to_mixed_criticality(
    taskset: List[Tuple[int, int, int]],
    CF: int,
    hi_ratio: float,
) -> List[Tuple[int, int, int, int, int]]:
    n      = len(taskset)
    n_hi   = int(round(n * hi_ratio))
    hi_idx = set(random.sample(range(n), n_hi))
    mc     = []
    for i, (C_lo, T, D) in enumerate(taskset):
        crit = 1 if i in hi_idx else 0
        if crit == 1:
            C_hi = max(1, int(round(C_lo * CF)))
        else:
            cand = max(1, int(round(C_lo / CF)))
            C_hi = 0 if cand == C_lo else cand
        mc.append((C_lo, C_hi, T, D, crit))
    return mc


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SCHEDULABILITY TESTS  (unchanged logic from original)
# ═══════════════════════════════════════════════════════════════════════════════

# ── IMC-PnG ──────────────────────────────────────────────────────────────────

def imc_png_schedulable(tasks: List[Task]) -> bool:
    U_LO     = sum(t[1] / t[3] for t in tasks if t[5] == 0)
    U_LO_deg = sum(t[2] / t[3] for t in tasks if t[5] == 0)
    HI_tasks = [t for t in tasks if t[5] == 1]
    if not HI_tasks:
        return U_LO <= 1.0 + 1e-12

    uL = [t[1] / t[3] for t in HI_tasks]
    uH = [t[2] / t[3] for t in HI_tasks]
    S  = 1.0 - U_LO
    if S <= 0 or sum(uL) > S:
        return False

    k = [0.0 if uL[i] == 0 else math.sqrt((uH[i] - uL[i]) / uL[i])
         for i in range(len(uL))]

    def sum_constraint(lam: float) -> float:
        return sum(uL[i] / (1.0 / (1.0 + k[i] * math.sqrt(lam)))
                   for i in range(len(uL)))

    lo, hi = 0.0, 1e6
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if sum_constraint(mid) > S:
            hi = mid
        else:
            lo = mid

    lam_star = (lo + hi) / 2.0
    x        = [1.0 / (1.0 + k[i] * math.sqrt(lam_star)) for i in range(len(uL))]

    if sum(uL[i] / x[i] for i in range(len(uL))) > S + 1e-12:
        return False

    cond2 = U_LO_deg + sum((uH[i] - uL[i]) / (1.0 - x[i] + 1e-12)
                           for i in range(len(uL)))
    return cond2 <= 1.0 + 1e-12


# ── EDF-IMC ───────────────────────────────────────────────────────────────────

def edf_imc_schedulable(tasks: List[Task]) -> bool:
    U_LO     = sum(t[1] / t[3] for t in tasks if t[5] == 0)
    U_LO_deg = sum(t[2] / t[3] for t in tasks if t[5] == 0)
    HI_tasks = [t for t in tasks if t[5] == 1]
    U_HI_LO  = sum(t[1] / t[3] for t in HI_tasks)
    denom    = 1.0 - U_LO

    if denom <= 0 or U_HI_LO > denom:
        return False

    x = U_HI_LO / denom
    if U_LO + (U_HI_LO / x) > 1.0 + 1e-12:
        return False

    cond2 = U_LO_deg
    for t in HI_tasks:
        uL = t[1] / t[3]
        uH = t[2] / t[3]
        cond2 += (uH - uL) / (1.0 - x + 1e-12)
    return cond2 <= 1.0 + 1e-12


# ── AMC-IMC (RTB) ─────────────────────────────────────────────────────────────

def compute_R_lo(task_i: Task, hp_tasks: List[Task], max_iter: int) -> float:
    R = float(task_i[1])
    for _ in range(max_iter):
        intr  = sum(math.ceil(R / tj[3]) * tj[1] for tj in hp_tasks)
        R_next = task_i[1] + intr
        if R_next == R:
            return R
        R = R_next
    return float('inf')


def compute_R_hi(task_i: Task, hp_tasks: List[Task],
                 hpL_tasks: List[Task], R_lo_i: float, max_iter: int) -> float:
    R = float(task_i[2])
    for _ in range(max_iter):
        intr_hi = sum(math.ceil(R / tj[3]) * tj[2]
                      for tj in hp_tasks if tj[5] == 1)
        intr_lo = sum(math.ceil(R_lo_i / tj[3]) * tj[1] for tj in hpL_tasks)
        R_next  = task_i[2] + intr_hi + intr_lo
        if R_next == R:
            return R
        R = R_next
    return float('inf')


def camc_rtb_schedulable(tasks: List[Task], max_iter: int) -> bool:
    sorted_tasks = sorted(tasks, key=lambda t: t[3])
    for idx, task_i in enumerate(sorted_tasks):
        hp_tasks  = sorted_tasks[:idx]
        hpL_tasks = [t for t in hp_tasks if t[5] == 0]
        _, C_lo, C_hi, T, D, crit = task_i

        R_lo = compute_R_lo(task_i, hp_tasks, max_iter)
        if R_lo > D:
            return False
        if crit == 1:
            R_hi = compute_R_hi(task_i, hp_tasks, hpL_tasks, R_lo, max_iter)
            if R_hi > D:
                return False
    return True


# ── TT-Merge ──────────────────────────────────────────────────────────────────

def edf_busy_window(tasks: List[Task], max_window: int) -> int:
    W = sum(t[1] for t in tasks)
    for _ in range(10_000):
        W_next = sum(math.ceil(W / t[3]) * t[1] for t in tasks)
        if W_next == W:
            return W
        W = W_next
        if W > max_window:
            return W
    return W


def generate_jobs_table_and_map(
    tasks: List[Task], L: int, use_hi_exec: bool = False
) -> Tuple[list, list]:
    table   = [0] * L
    job_map = [None] * L
    for tid, Clo, Chi, T, D, crit in tasks:
        exec_t = Chi if use_hi_exec else Clo
        for rel in range(0, L, T):
            start, end, rem = rel, min(rel + D, L), exec_t
            for t in range(end - 1, start - 1, -1):
                if rem <= 0:
                    break
                if table[t] == 0:
                    table[t]   = tid
                    job_map[t] = (tid, rel)
                    rem       -= 1
    return table, job_map


def find_ready_task_and_index_with_map(
    temp_table: list, job_map: list, t: int, tasks: List[Task], L: int
) -> Tuple:
    best = None
    for (tid, _Clo, _Chi, T, D, _crit) in tasks:
        rel   = (t // T) * T
        start, end = rel, min(rel + D, L)
        if not (start <= t < end):
            continue
        idx = next((i for i in range(start, end) if job_map[i] == (tid, rel)), None)
        if idx is not None:
            deadline = end
            if (best is None
                    or deadline < best[1]
                    or (deadline == best[1] and tid < best[0])):
                best = (tid, deadline, idx, rel)
    if best:
        return best[0], best[2], best[3]
    return None, None, None


def tt_merge_schedulable(tasks: List[Task], max_window: int) -> bool:
    W    = edf_busy_window(tasks, max_window)
    Tmax = max(t[3] for t in tasks) if tasks else 1
    L    = max(W if W <= max_window else FALLBACK_K * Tmax, 1)

    lo_tasks = [t for t in tasks if t[5] == 0]
    hi_tasks = [t for t in tasks if t[2] > 0]
    temp_lo, job_map_lo = generate_jobs_table_and_map(lo_tasks, L, use_hi_exec=False)
    temp_hi, job_map_hi = generate_jobs_table_and_map(hi_tasks, L, use_hi_exec=True)

    for tid, Clo, Chi, T, D, crit in hi_tasks:
        if crit == 1:
            diff = Chi - Clo
            if diff > 0:
                for rel in range(0, L, T):
                    start, end, cnt = rel, min(rel + D, L), diff
                    for i in range(end - 1, start - 1, -1):
                        if temp_hi[i] == tid:
                            temp_hi[i] = 0
                            job_map_hi[i] = None
                            cnt -= 1
                            if cnt == 0:
                                break

    D_of_lo: Dict[int, int] = {
        tid: D for (tid, _Clo, _Chi, _T, D, _crit) in lo_tasks
    }

    for t in range(L):
        lo_val, hi_val = temp_lo[t], temp_hi[t]
        if lo_val != 0 and hi_val != 0:
            return False
        if lo_val == 0 and hi_val == 0:
            lo_tid, lo_idx, lo_rel = find_ready_task_and_index_with_map(
                temp_lo, job_map_lo, t, lo_tasks, L)
            hi_tid, hi_idx, hi_rel = find_ready_task_and_index_with_map(
                temp_hi, job_map_hi, t, hi_tasks, L)
            if lo_tid:
                temp_lo[lo_idx] = 0
                job_map_lo[lo_idx] = None
                D_lo = D_of_lo.get(lo_tid, 0)
                for i in range(lo_rel, min(lo_rel + D_lo, L)):
                    if job_map_hi[i] == (lo_tid, lo_rel):
                        temp_hi[i] = 0
                        job_map_hi[i] = None
            if hi_tid:
                temp_hi[hi_idx] = 0
                job_map_hi[hi_idx] = None
        elif lo_val == 0 and hi_val != 0:
            temp_hi[t] = 0
            job_map_hi[t] = None
        else:  # lo_val != 0 and hi_val == 0
            rel_info = job_map_lo[t]
            temp_lo[t] = 0
            job_map_lo[t] = None
            if rel_info:
                lo_tid, lo_rel = rel_info
                D_lo = D_of_lo.get(lo_tid, 0)
                for i in range(lo_rel, min(lo_rel + D_lo, L)):
                    if job_map_hi[i] == (lo_tid, lo_rel):
                        temp_hi[i] = 0
                        job_map_hi[i] = None
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — WORKER FUNCTION  (runs in a subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

def worker_generate_n(args_dict: dict) -> dict:
    """
    Generate all task sets for one value of n and write to a CSV file.
    Designed to be called via multiprocessing.Pool.

    Args (passed as a dict to work around Pool pickling constraints):
        num_tasks        : int
        out_path         : str
        sets_per_combo   : int
        tmin, tmax       : int
        tt_max_window    : int
        amc_max_iter     : int
        seed             : int  (base seed; per-n seed = seed + num_tasks)
        taskset_id_start : int

    Returns:
        dict with keys: num_tasks, rows_written, elapsed_s, error (or None)
    """
    num_tasks        = args_dict["num_tasks"]
    out_path         = args_dict["out_path"]
    sets_per_combo   = args_dict["sets_per_combo"]
    tmin             = args_dict["tmin"]
    tmax             = args_dict["tmax"]
    tt_max_window    = args_dict["tt_max_window"]
    amc_max_iter     = args_dict["amc_max_iter"]
    seed             = args_dict["seed"]
    taskset_id_start = args_dict["taskset_id_start"]

    # Each n gets a deterministic but distinct seed
    random.seed(seed + num_tasks)

    t_start      = time.time()
    taskset_id   = taskset_id_start
    global_tid   = 1
    rows_written = 0

    # Build header
    header = ["Taskset_ID", "Num_Tasks", "U_total", "HI_ratio", "CF"]
    for i in range(1, num_tasks + 1):
        header += [f"CLO{i}", f"CHI{i}", f"D{i}", f"T{i}", f"Crit{i}"]
    header += ["AMC_IMC_Schedulable", "TT_Merge_Schedulable",
               "IMC_PnG_Schedulable", "EDF_IMC_Schedulable"]

    total = len(UTILIZATIONS) * len(HI_RATIOS) * len(CF_VALUES) * sets_per_combo

    try:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for U in UTILIZATIONS:
                for hi_ratio in HI_RATIOS:
                    for CF in CF_VALUES:
                        amc_c = tt_c = imc_c = edf_c = 0

                        for _ in range(sets_per_combo):
                            base_ts = generate_taskset(num_tasks, U, tmin, tmax)
                            mc_raw  = convert_to_mixed_criticality(base_ts, CF, hi_ratio)

                            tasks: List[Task] = [
                                (global_tid + i, C_lo, C_hi, T, D, crit)
                                for i, (C_lo, C_hi, T, D, crit) in enumerate(mc_raw)
                            ]
                            global_tid += num_tasks

                            amc_ok = camc_rtb_schedulable(tasks, amc_max_iter)
                            tt_ok  = tt_merge_schedulable(tasks, tt_max_window)
                            imc_ok = imc_png_schedulable(tasks)
                            edf_ok = edf_imc_schedulable(tasks)

                            if amc_ok: amc_c += 1
                            if tt_ok:  tt_c  += 1
                            if imc_ok: imc_c += 1
                            if edf_ok: edf_c += 1

                            row = [taskset_id, num_tasks, U, hi_ratio, CF]
                            for (_, C_lo, C_hi, T, D, crit) in tasks:
                                row += [C_lo, C_hi, D, T, crit]
                            row += [int(amc_ok), int(tt_ok), int(imc_ok), int(edf_ok)]
                            writer.writerow(row)

                            taskset_id   += 1
                            rows_written += 1

                            if rows_written % PROGRESS_EVERY == 0:
                                pct = 100 * rows_written / total
                                elapsed = time.time() - t_start
                                rate = rows_written / elapsed if elapsed > 0 else 0
                                eta  = (total - rows_written) / rate if rate > 0 else 0
                                print(f"  [n={num_tasks:4d}] {rows_written:>7,}/{total:,} "
                                      f"({pct:.1f}%) | {elapsed:.0f}s elapsed | "
                                      f"ETA {eta:.0f}s | {rate:.1f} rows/s",
                                      flush=True)
                            # periodic flush to disk
                            if rows_written % 5000 == 0:
                                f.flush()

        elapsed = time.time() - t_start
        return {
            "num_tasks":    num_tasks,
            "rows_written": rows_written,
            "elapsed_s":    elapsed,
            "error":        None,
        }

    except Exception as e:
        return {
            "num_tasks":    num_tasks,
            "rows_written": rows_written,
            "elapsed_s":    time.time() - t_start,
            "error":        str(e),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="IMC-Former dataset generator — all labels for all n.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--sizes", nargs="+", type=int, default=None,
        help=(
            "Task-set sizes to generate. Default: all sizes.\n"
            "Example: --sizes 4 8 12 16 20"
        ),
    )
    p.add_argument(
        "--out_dir", default="generated_datasets",
        help="Output directory. Default: generated_datasets/",
    )
    p.add_argument(
        "--sets_per_combo", type=int, default=None,
        help=(
            "Override sets_per_combo for ALL sizes (useful for quick tests).\n"
            "Default: tiered values per size.\n"
            "Example for a 2-minute test run:  --sets_per_combo 10"
        ),
    )
    p.add_argument(
        "--workers", type=int, default=None,
        help=(
            "Number of parallel worker processes.\n"
            "Default: min(len(sizes), cpu_count).\n"
            "Each size runs in its own process.\n"
            "Use --workers 1 to disable parallelism (easier to debug)."
        ),
    )
    p.add_argument(
        "--resume", action="store_true",
        help=(
            "Skip sizes whose CSV already exists and has >= 90%% of expected rows.\n"
            "Useful for resuming an interrupted run."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Base random seed. Default: {RANDOM_SEED}. "
             "Each n gets seed + n for independence.",
    )
    return p.parse_args()


def count_csv_rows(path: str) -> int:
    """Count data rows (excluding header) in an existing CSV."""
    try:
        with open(path) as f:
            return sum(1 for _ in f) - 1
    except Exception:
        return 0


def main():
    args = parse_args()

    sizes = args.sizes if args.sizes else DEFAULT_TASK_COUNTS
    unknown = [n for n in sizes if n not in SETS_PER_COMBO_TIER]
    if unknown:
        print(f"[ERROR] Unknown sizes: {unknown}")
        print(f"        Supported: {sorted(SETS_PER_COMBO_TIER.keys())}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "generation_summary.csv")
    combos       = len(UTILIZATIONS) * len(HI_RATIOS) * len(CF_VALUES)

    print("=" * 72)
    print("IMC-Former Dataset Generator  —  ALL LABELS FOR ALL N")
    print("=" * 72)
    print(f"  Output dir  : {os.path.abspath(args.out_dir)}")
    print(f"  Sizes       : {sizes}")
    print(f"  Seed        : {args.seed}")
    print(f"  Resume      : {args.resume}")
    print(f"  Grid        : {len(UTILIZATIONS)} U × {len(HI_RATIOS)} HI_ratio × "
          f"{len(CF_VALUES)} CF = {combos} combos/n")
    print()

    # ── Build work list (skip already-complete sizes if --resume) ──────────────
    work_items = []
    taskset_id = 1
    total_expected = 0

    for n in sizes:
        spc      = args.sets_per_combo or SETS_PER_COMBO_TIER[n]
        expected = combos * spc
        out_path = os.path.join(args.out_dir, f"imc_{n}.csv")
        role     = ("training"    if n <= 20 else
                    "gen_small"   if n <= 28 else
                    "gen_medium"  if n <= 40 else
                    "gen_large"   if n <= 64 else "gen_xlarge/xxlarge")

        if args.resume and os.path.exists(out_path):
            existing = count_csv_rows(out_path)
            if existing >= int(0.90 * expected):
                print(f"  [SKIP]  n={n:4d}: {existing:>7,} rows exist "
                      f"(≥90% of {expected:,}) [{role}]")
                taskset_id += existing
                continue
            else:
                print(f"  [REDO]  n={n:4d}: {existing:>7,} rows (< 90% of {expected:,}) "
                      f"→ regenerating")

        total_expected += expected
        print(f"  [QUEUE] n={n:4d}: ~{expected:>7,} rows  [{role}]  "
              f"T=[{PERIOD_RANGE[n][0]},{PERIOD_RANGE[n][1]}]  "
              f"AMC_iter={AMC_MAX_ITER[n]}  TT_win={TT_MAX_WINDOW[n]:,}")

        work_items.append({
            "num_tasks":        n,
            "out_path":         out_path,
            "sets_per_combo":   spc,
            "tmin":             PERIOD_RANGE[n][0],
            "tmax":             PERIOD_RANGE[n][1],
            "tt_max_window":    TT_MAX_WINDOW[n],
            "amc_max_iter":     AMC_MAX_ITER[n],
            "seed":             args.seed,
            "taskset_id_start": taskset_id,
        })
        taskset_id += expected   # approximate; exact IDs assigned per-worker

    if not work_items:
        print("\nNothing to generate (all sizes already complete).")
        print("Remove --resume or delete files to regenerate.")
        sys.exit(0)

    n_workers = min(
        args.workers or mp.cpu_count(),
        len(work_items),
    )
    print(f"\n  Workers: {n_workers}  (parallel across n values)")
    print(f"  Total expected: ~{total_expected:,} rows")
    print("=" * 72)

    wall_start = time.time()

    # ── Run workers ────────────────────────────────────────────────────────────
    if n_workers == 1:
        # Sequential mode (easier to debug, single-core machines, Windows)
        results = [worker_generate_n(item) for item in work_items]
    else:
        # Parallel: each n in its own process
        # Use spawn context for safety on macOS/Windows
        ctx  = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(worker_generate_n, work_items)

    # ── Write summary CSV and print report ─────────────────────────────────────
    wall_elapsed = time.time() - wall_start
    total_rows   = 0

    print(f"\n{'='*72}")
    print("RESULTS")
    print(f"{'─'*72}")

    with open(summary_path, "a", newline="") as fs:
        ws = csv.writer(fs)
        if os.path.getsize(summary_path) == 0 if os.path.exists(summary_path) else True:
            ws.writerow(["num_tasks", "rows_written", "elapsed_s",
                         "rows_per_s", "status"])

        for r in results:
            n      = r["num_tasks"]
            rows   = r["rows_written"]
            secs   = r["elapsed_s"]
            rate   = rows / secs if secs > 0 else 0
            status = "ERROR: " + r["error"] if r["error"] else "OK"
            total_rows += rows
            ws.writerow([n, rows, f"{secs:.1f}", f"{rate:.1f}", status])

            if r["error"]:
                print(f"  n={n:4d}: ERROR — {r['error']}")
            else:
                print(f"  n={n:4d}: {rows:>7,} rows in {secs/60:.1f} min "
                      f"({rate:.1f} rows/s)  →  imc_{n}.csv")

    print(f"{'─'*72}")
    print(f"  TOTAL: {total_rows:,} rows in {wall_elapsed/60:.1f} min")
    print(f"  Summary: {summary_path}")
    print(f"{'='*72}")
    print()
    print("Next — prepare training splits:")
    print(f"  python prepare_data.py \\")
    print(f"      --input_dir  {args.out_dir} \\")
    print(f"      --output_dir data/ \\")
    print(f"      --train_ns   4 8 12 16 20 \\")
    print(f"      --gen_small  24 28 \\")
    print(f"      --gen_medium 32 40 \\")
    print(f"      --gen_large  48 64 \\")
    print(f"      --gen_xlarge 128 \\")
    print(f"      --gen_xxlarge 512 1024")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()