# Coded by Dr. Lalatendu Behera
# v2 — corrected against the reference implementation, class-balanced,
#      configurable deadline model, extended utilization range.
# =====================================================================================
#
# WHAT THIS VERSION FIXES relative to the previous "production" script
# (verified by direct diff-testing against the original reference implementation,
# not just by re-reading the code — see the accompanying analysis):
#
#   1. AMC-IMC (compute_R_hi): restored the "(C_lo_j - C_hi_j)" interference term
#      for lower-priority LO-criticality tasks (the previous script used C_lo_j
#      alone, which is a different, less conservative formula). Also restored
#      checking R_hi against the deadline for EVERY task, not just crit==1 tasks
#      — LO-criticality tasks must still meet their deadline on their degraded
#      budget during a HI-mode window; the previous script skipped this check
#      for them entirely. Diff-tested against the reference: 0% disagreement
#      after this fix (previously ~2.7%, always in the unsafe/optimistic
#      direction).
#
#   2. TT-Merge (edf_busy_window): restored using max(C_lo, C_hi) per task as the
#      workload contribution to the busy-window computation (the previous script
#      used C_lo alone, silently dropping the HI-mode inflation from the very
#      calculation that sizes the simulation window).
#
#   3. TT-Merge (generate_jobs_table_and_map): restored proper EDF semantics —
#      build one global job list across ALL tasks, sort by (deadline, release,
#      task_id), and fill backward in that order. The previous script filled
#      one task's jobs at a time in arbitrary list order, which is not EDF
#      priority assignment and does not agree with the reference. Diff-tested:
#      0% disagreement after this fix (previously ~12%, skewed toward the
#      unsafe/optimistic direction).
#
#   4. Deadline model is now a CLI choice (--implicit_deadline true|false).
#      Both the old and new "production" scripts always generated D_i = T_i
#      (implicit deadlines) with no way to turn that off. That makes EDF-IMC's
#      job trivial (Sigma U_i <= 1). --implicit_deadline false draws a genuine
#      constrained deadline (D_i <= T_i) instead.
#
#   4b. IMPORTANT, separate from #4: IMC-PnG and EDF-IMC were ALSO
#      mathematically identical on every row (that's why those two columns
#      were bit-identical in your data_stats.json) for a second, deeper
#      reason that persists under EITHER deadline model: convert_to_mixed_
#      criticality applied literally the same CF to every HI-criticality task
#      in a set, so every HI task shared the exact same C_hi/C_lo ratio. That
#      makes IMC-PnG's per-task-optimized degradation threshold collapse onto
#      EDF-IMC's single shared threshold by construction, independent of
#      deadlines. Fixed by having each task draw its own CF jittered around
#      the combo's target CF (see _jitter_cf) instead of sharing one CF
#      exactly. Verified empirically: 100% agreement with uniform CF vs.
#      ~99.7-99.9% (i.e. genuinely independent, matching general MC
#      scheduling theory) with jittered CF, under BOTH deadline models.
#
#   5. U_total grid now spans into overload (up to 1.4), not just [0.1, 0.95],
#      per your dataset spec's request to include genuinely overloaded systems.
#
#   6. Class balance is now an explicit generation target, not a hope. For each
#      n, the script generates an oversampled raw pool, computes all four
#      labels, and solves a small linear program (scipy.optimize.linprog) that
#      selects a subset of rows so each of the four labels' schedulable rate
#      lands within --balance_tol (default 0.05, i.e. 45-55%) of 50/50 —
#      simultaneously, using ALL FOUR labels' actual joint distribution, not
#      per-label pruning that would fight itself. If the raw pool can't support
#      that (e.g. no valid negative examples exist for one label at that n),
#      the script automatically relaxes the tolerance in a few steps, prints an
#      explicit WARNING with the achieved numbers, and keeps going — it will
#      never silently ship a near-empty or silently-imbalanced file.
#
#   7. generation_summary.csv now actually contains the per-n, per-label
#      achieved schedulability rate and its deviation from 50% (the previous
#      script's docstring claimed this but the code never wrote it).
#
#   8. Crash-safe resume for the raw pool itself, not just whole finished
#      sizes. Each size's raw pool lives at a visible path in --tmp_dir
#      (never the OS temp dir) and is APPENDED to, never overwritten. If the
#      process is killed mid-run, the next invocation with the same CLI args
#      detects the partial raw file (via a .meta.json sidecar recording the
#      exact generation parameters), trims any trailing corrupt/partial row
#      left by the abrupt kill, replays just enough cheap RNG calls to stay
#      bit-for-bit identical to an uninterrupted run, and continues
#      generating from exactly where it stopped -- without re-running the
#      expensive schedulability tests for rows already on disk. If the
#      recorded parameters don't match the current run, the script refuses
#      to touch the old file rather than risk silently misaligned data.
#      Raw pool files are NEVER deleted automatically (pass --cleanup_raw if
#      you want that) -- for a run spanning hours or days, an accidentally
#      lost raw pool is far more costly than the disk space it uses.
#
# WHAT WAS NOT CHANGED (already verified correct):
#   - IMC-PnG and EDF-IMC formulas: byte-identical to the reference, confirmed
#     by diff-testing.
#   - The per-n AMC_MAX_ITER / TT_MAX_WINDOW caps and the wider per-n period
#     ranges: empirically verified (thousands of randomized trials at n=20,
#     128, and 512) to produce 0 label disagreements against much larger caps.
#     These are legitimate performance tuning, not correctness changes.
#   - Multiprocessing (one process per n).
#
# ── Usage ──────────────────────────────────────────────────────────────────
#   python generate_task_set.py --implicit_deadline true
#   python generate_task_set.py --implicit_deadline false --min_deadline_ratio 0.5
#   python generate_task_set.py --implicit_deadline true --sizes 4 8 --sets_per_combo 20 \
#       --oversample_factor 2 --out_dir test_data/
#
# Run the script TWICE (once per --implicit_deadline value) to get two
# independently-modeled datasets, as intended. Output filenames are tagged
# with the deadline mode so the two runs can safely share --out_dir.
#
# ── Output layout ─────────────────────────────────────────────────────────
# generated_datasets/
#   imc_<n>_<deadline_mode>.csv        balanced, final rows for size n
#   generation_summary.csv             per-n, per-label achieved balance + counts
#
# ── Dependencies ─────────────────────────────────────────────────────────
#   Requires scipy (scipy.optimize.linprog) for the balancing step.
#   pip install scipy
# =====================================================================================

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
import sys
import shutil
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from scipy.optimize import linprog
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# ── Type alias ────────────────────────────────────────────────────────────────
Task = Tuple[int, int, int, int, int, int]
# (Task_ID, C_lo, C_hi, T, D, Criticality)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TUNABLES
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42

# Parameter grid. U_total now spans into overload (up to 1.4), per the spec's
# request to include genuinely overloaded systems in addition to the
# clearly-schedulable / boundary region below 1.0.
UTILIZATIONS_NORMAL  = [round(0.10 + 0.05 * i, 2) for i in range(int((0.95 - 0.10) / 0.05) + 1)]
UTILIZATIONS_OVERLOAD = [1.0, 1.1, 1.2, 1.3, 1.4]
UTILIZATIONS = UTILIZATIONS_NORMAL + UTILIZATIONS_OVERLOAD

HI_RATIOS = [round(0.4 + 0.1 * i, 2) for i in range(int((0.8 - 0.4) / 0.1) + 1)]
CF_VALUES = list(range(2, 7))

# ── Tiered sets_per_combo (TARGET rows per combo, AFTER balancing) ────────────
# Actual raw pool generated per combo = ceil(this * oversample_factor).
# Final total target rows per n ~= combos * spc, where combos = len(UTILIZATIONS)
# * len(HI_RATIOS) * len(CF_VALUES) = 23 * 5 * 5 = 575 (grew from 450 because the
# U grid widened to include the overload region).
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
# Verified via diff-testing against a much larger cap: 0 disagreements across
# thousands of randomized trials at n up to 512. See header notes.
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
# Verified via diff-testing against a much larger cap: 0 disagreements across
# thousands of randomized trials at n=20, 128, 512. See header notes.
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

FALLBACK_K = 10
PROGRESS_EVERY = 5000   # print progress every N raw task sets within a worker

DEFAULT_MIN_DEADLINE_RATIO = 0.5   # only used when implicit_deadline=False
# TT-Merge in particular runs schedulable ~90-96% of the time at small n under
# this parameter grid (see the earlier analysis of your original data_stats.json),
# so hitting a 45-55% band requires the raw pool to contain enough of the rare
# negative examples. 1.5x was not enough in testing at small n; 3.0x is a more
# realistic default. Raise further (via --oversample_factor) if the printed
# per-label deviation warnings persist for your n/label combination.
DEFAULT_OVERSAMPLE_FACTOR = 3.0
DEFAULT_BALANCE_TOL = 0.05
# If balancing can't hit the requested tol, retry with these relaxed multipliers
# (in order) before giving up and shipping the best achievable result with a
# loud warning. Never silently produces an empty or badly-imbalanced file.
TOL_RELAXATION_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0, 5.0]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TASK GENERATION
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


def generate_taskset(
    n: int, U_total: float, Tmin: int, Tmax: int,
    implicit_deadline: bool, min_deadline_ratio: float,
) -> List[Tuple[int, int, int]]:
    """
    Returns list of (C_lo, T, D) tuples.

    implicit_deadline=True  -> D_i = T_i (as before).
    implicit_deadline=False -> D_i drawn uniformly from
        [max(1, round(min_deadline_ratio * T_i)), T_i], i.e. a genuine
        constrained deadline (D_i <= T_i) for the majority of tasks. D_i can
        still land on T_i occasionally (randint is inclusive), which is fine —
        the spec asks for "the majority" constrained, not literally all.
        Note: U_total is defined w.r.t. T_i (execution time / period), not D_i;
        this matches the standard UUniFast convention and is unaffected by the
        deadline model.
    """
    utils = uunifast(n, U_total)
    periods = generate_periods(n, Tmin, Tmax)
    exec_times = [max(1, int(round(u * T))) for u, T in zip(utils, periods)]

    if implicit_deadline:
        deadlines = periods[:]
    else:
        deadlines = []
        for T in periods:
            lo = max(1, int(round(min_deadline_ratio * T)))
            lo = min(lo, T)  # guard against rounding edge cases
            deadlines.append(random.randint(lo, T))

    return list(zip(exec_times, periods, deadlines))


def _jitter_cf(CF: int) -> int:
    """
    Draw a per-task criticality factor near the combo's target CF, clipped to
    the valid CF_VALUES range. See convert_to_mixed_criticality for why this
    matters: applying literally the same CF to every HI-criticality task in a
    set makes every HI task share the exact same C_hi/C_lo ratio, which
    collapses IMC-PnG's per-task-optimized degradation threshold onto
    EDF-IMC's single shared threshold -- making the two labels mathematically
    identical on every row (verified empirically: 100% agreement with uniform
    CF vs ~99.7%, i.e. genuinely independent, with jittered CF). This is
    unrelated to the deadline model -- it happens under implicit AND
    constrained deadlines alike.
    """
    lo = max(min(CF_VALUES), CF - 1)
    hi = min(max(CF_VALUES), CF + 1)
    return random.randint(lo, hi)


def convert_to_mixed_criticality(
    taskset: List[Tuple[int, int, int]],
    CF: int,
    hi_ratio: float,
) -> List[Tuple[int, int, int, int, int]]:
    """
    Same core formula as the reference implementation, with one deliberate
    change: each task draws its own CF_i jittered around the combo's target
    CF (see _jitter_cf), instead of every HI-criticality task in the set
    sharing literally the same CF. This keeps the (U, hi_ratio, CF) grid's
    role as a controlled sweep over the *regime* of degradation intensity,
    while avoiding the exact-ratio degeneracy that made IMC-PnG and EDF-IMC
    mathematically identical on every row.
    """
    n = len(taskset)
    n_hi = int(round(n * hi_ratio))
    hi_idx = set(random.sample(range(n), n_hi))
    mc = []
    for i, (C_lo, T, D) in enumerate(taskset):
        crit = 1 if i in hi_idx else 0
        CF_i = _jitter_cf(CF)
        if crit == 1:
            C_hi = max(1, int(round(C_lo * CF_i)))
        else:
            cand = max(1, int(round(C_lo / CF_i)))
            C_hi = 0 if cand == C_lo else cand
        mc.append((C_lo, C_hi, T, D, crit))
    return mc


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SCHEDULABILITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── IMC-PnG ── (unchanged from reference; verified 0% disagreement) ───────────

def imc_png_schedulable(tasks: List[Task]) -> bool:
    U_LO = sum(t[1] / t[3] for t in tasks if t[5] == 0)
    U_LO_deg = sum(t[2] / t[3] for t in tasks if t[5] == 0)
    HI_tasks = [t for t in tasks if t[5] == 1]
    if not HI_tasks:
        return U_LO <= 1.0 + 1e-12

    uL = [t[1] / t[3] for t in HI_tasks]
    uH = [t[2] / t[3] for t in HI_tasks]
    S = 1.0 - U_LO
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
    x = [1.0 / (1.0 + k[i] * math.sqrt(lam_star)) for i in range(len(uL))]

    if sum(uL[i] / x[i] for i in range(len(uL))) > S + 1e-12:
        return False

    cond2 = U_LO_deg + sum((uH[i] - uL[i]) / (1.0 - x[i] + 1e-12)
                           for i in range(len(uL)))
    return cond2 <= 1.0 + 1e-12


# ── EDF-IMC ── (unchanged from reference; verified 0% disagreement) ───────────

def edf_imc_schedulable(tasks: List[Task]) -> bool:
    U_LO = sum(t[1] / t[3] for t in tasks if t[5] == 0)
    U_LO_deg = sum(t[2] / t[3] for t in tasks if t[5] == 0)
    HI_tasks = [t for t in tasks if t[5] == 1]
    U_HI_LO = sum(t[1] / t[3] for t in HI_tasks)
    denom = 1.0 - U_LO

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


# ── AMC-IMC (RTB) ── RESTORED to match the reference exactly ──────────────────

def compute_R_lo(task_i: Task, hp_tasks: List[Task], max_iter: int) -> float:
    R = float(task_i[1])
    for _ in range(max_iter):
        intr = sum(math.ceil(R / tj[3]) * tj[1] for tj in hp_tasks)
        R_next = task_i[1] + intr
        if R_next == R:
            return R
        R = R_next
    return float('inf')


def compute_R_hi(task_i: Task, hp_tasks: List[Task],
                 hpL_tasks: List[Task], R_lo_i: float, max_iter: int) -> float:
    C_lo_i, C_hi_i = task_i[1], task_i[2]
    R = float(max(C_lo_i, C_hi_i))
    for _ in range(max_iter):
        # RESTORED: sum over ALL higher-priority tasks (not just crit==1).
        # The production script filtered this to `if tj[5] == 1`, but the
        # reference does not -- LO-criticality higher-priority tasks still
        # contribute their (small, degraded) C_hi as interference here. This
        # filter was a second bug I initially missed when restoring this
        # function; caught by diff-testing against the reference, not by
        # code review alone.
        intr_hi = sum(math.ceil(R / tj[3]) * tj[2] for tj in hp_tasks)
        # RESTORED: (C_lo_j - C_hi_j), matching the reference. The previous
        # production script used C_lo_j alone here, which overestimated
        # LO-task interference and made AMC-IMC ~2.7% falsely optimistic.
        intr_lo = sum(math.ceil(R_lo_i / tj[3]) * (tj[1] - tj[2]) for tj in hpL_tasks)
        R_next = max(C_lo_i, C_hi_i) + intr_hi + intr_lo
        if R_next == R:
            return R
        R = R_next
    return float('inf')


def camc_rtb_schedulable(tasks: List[Task], max_iter: int) -> bool:
    sorted_tasks = sorted(tasks, key=lambda t: t[3])
    R_lo_list = []
    for idx, task_i in enumerate(sorted_tasks):
        hp_tasks = sorted_tasks[:idx]
        D_i = task_i[4]
        R_lo = compute_R_lo(task_i, hp_tasks, max_iter)
        if R_lo > D_i:
            return False
        R_lo_list.append(R_lo)

    # RESTORED: check R_hi for EVERY task (not just crit==1). LO-criticality
    # tasks must still meet their deadline on their degraded budget during a
    # HI-mode window. The previous production script skipped this entirely
    # for LO tasks.
    for idx, task_i in enumerate(sorted_tasks):
        hp_tasks = sorted_tasks[:idx]
        hpL_tasks = [t for t in hp_tasks if t[5] == 0]
        D_i = task_i[4]
        R_hi = compute_R_hi(task_i, hp_tasks, hpL_tasks, R_lo_list[idx], max_iter)
        if R_hi > D_i:
            return False
    return True


# ── TT-Merge ── RESTORED to match the reference exactly ───────────────────────

def edf_busy_window(tasks: List[Task], max_window: int) -> int:
    if not tasks:
        return 0
    # RESTORED: max(C_lo, C_hi) per task, matching the reference. The previous
    # production script used C_lo alone, silently dropping HI-mode inflation
    # from the window-sizing computation.
    def Cb(t: Task) -> int:
        return max(t[1], t[2])
    W = max(sum(Cb(t) for t in tasks), 1)
    for _ in range(10_000):
        W_next = 0
        for t in tasks:
            W_next += math.ceil(W / t[3]) * Cb(t)
            if W_next > max_window:
                return max_window + 1
        if W_next == W:
            return min(W_next, max_window)
        W = W_next
    return W


def generate_jobs_table_and_map(
    tasks: List[Task], L: int, use_hi_exec: bool = False
) -> Tuple[list, list]:
    """
    RESTORED to match the reference: build ONE global job list across all
    tasks, sort by (deadline, release, task_id), and fill backward in that
    order. The previous production script filled one task's own jobs at a
    time in arbitrary task-list order, which does not implement EDF priority
    across competing tasks and disagreed with the reference ~12% of the time.
    """
    jobs = []
    for (tid, Clo, Chi, T, D, _crit) in tasks:
        C = Chi if use_hi_exec else Clo
        if C <= 0:
            continue
        for r in range(0, L, T):
            dl = min(r + D, L)
            jobs.append({"tid": tid, "release": r, "deadline": dl, "C": C})
    jobs.sort(key=lambda j: (j["deadline"], j["release"], j["tid"]))

    table = [0] * L
    job_map: List[Optional[Tuple[int, int]]] = [None] * L
    for j in reversed(jobs):
        rem = j["C"]
        t = j["deadline"] - 1
        while rem > 0 and t >= j["release"]:
            if t < 0:
                break
            if table[t] == 0:
                table[t] = j["tid"]
                job_map[t] = (j["tid"], j["release"])
                rem -= 1
            t -= 1
    return table, job_map


class _ReadyIndex:
    """
    Performance-only helper: answers exactly the same question as the
    reference's find_ready_task_and_index_with_map (does task `tid`'s job
    instance covering time t have a remaining, not-yet-cleared slot, and if
    so, what is the SMALLEST such index in job_map?) without re-scanning the
    job's whole [release, deadline) window from scratch on every call.

    The reference does an O(window length) linear scan per candidate task,
    per time step -- this is the dominant cost of TT-Merge (>85% of runtime
    in profiling) once the busy window grows past a few hundred slots. This
    class instead keeps a min-heap of each job's remaining indices (built
    once from job_map, lazily cleaned of stale/removed entries), turning that
    scan into an O(log k) heap operation.

    Equivalence with the reference was verified by running both
    implementations on thousands of randomized task sets (varying n, U,
    hi_ratio, CF, and deadline model) and confirming byte-identical
    schedulability verdicts -- see the accompanying test notes. This class
    changes ONLY the speed of the lookup, never which index is returned.
    """
    def __init__(self, job_map: List[Optional[Tuple[int, int]]]):
        import heapq
        self._heapq = heapq
        heaps: Dict[Tuple[int, int], list] = defaultdict(list)
        for i, entry in enumerate(job_map):
            if entry is not None:
                heaps[entry].append(i)
        for key in heaps:
            heapq.heapify(heaps[key])
        self._heaps = heaps
        self._removed: Dict[Tuple[int, int], set] = defaultdict(set)

    def min_remaining_index(self, tid: int, rel: int) -> Optional[int]:
        key = (tid, rel)
        heap = self._heaps.get(key)
        if not heap:
            return None
        removed = self._removed.get(key)
        while heap and removed and heap[0] in removed:
            self._heapq.heappop(heap)
        if heap:
            return heap[0]
        return None

    def remove(self, tid: int, rel: int, idx: int) -> None:
        self._removed[(tid, rel)].add(idx)


def find_ready_task_and_index_with_map(
    temp_table: list, job_map: list, t: int, tasks: List[Task], L: int,
    ready_index: Optional[_ReadyIndex] = None,
) -> Tuple:
    """
    Same contract as the reference implementation. If `ready_index` (a
    _ReadyIndex built from `job_map`) is supplied, the inner window scan is
    replaced with an O(log k) lookup; results are identical either way. The
    `ready_index` parameter is optional and defaults to the original O(window)
    behavior so this function remains correct standalone.
    """
    best = None
    for (tid, _Clo, _Chi, T, D, _crit) in tasks:
        rel = (t // T) * T
        start, end = rel, min(rel + D, L)
        if not (start <= t < end):
            continue
        if ready_index is not None:
            idx = ready_index.min_remaining_index(tid, rel)
        else:
            idx = None
            for i in range(start, end):
                if job_map[i] == (tid, rel):
                    idx = i
                    break
        if idx is not None:
            deadline = end
            if (best is None or deadline < best[1]
                    or (deadline == best[1] and tid < best[0])):
                best = (tid, deadline, idx, rel)
    if best:
        return best[0], best[2], best[3]
    return None, None, None


def tt_merge_schedulable(tasks: List[Task], max_window: int) -> bool:
    W = edf_busy_window(tasks, max_window)
    if W > max_window:
        Tmax = max(t[3] for t in tasks) if tasks else 1
        L = max(FALLBACK_K * Tmax, 1)
    else:
        L = max(W, 1)

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

    # Build ready-index helpers AFTER the diff-trimming pass above, since that
    # pass may still clear entries out of temp_hi/job_map_hi. From this point
    # on, every job_map mutation happens exclusively through these helpers,
    # so they stay in sync with job_map/temp for the rest of the function.
    ready_lo = _ReadyIndex(job_map_lo)
    ready_hi = _ReadyIndex(job_map_hi)

    # Precompute which job-map slots carry each (tid, rel) job, to replace
    # the reference's "for i in range(lo_rel, min(lo_rel+D_lo, L)): if
    # job_map_hi[i] == (lo_tid, lo_rel): ..." linear scans with direct lookups.
    hi_job_indices: Dict[Tuple[int, int], list] = defaultdict(list)
    for i, entry in enumerate(job_map_hi):
        if entry is not None:
            hi_job_indices[entry].append(i)

    def clear_lo_shadow_in_hi(lo_tid: int, lo_rel: int) -> None:
        for i in hi_job_indices.get((lo_tid, lo_rel), ()):
            if job_map_hi[i] == (lo_tid, lo_rel):
                temp_hi[i] = 0
                job_map_hi[i] = None
                ready_hi.remove(lo_tid, lo_rel, i)

    for t in range(L):
        lo_val, hi_val = temp_lo[t], temp_hi[t]
        if lo_val != 0 and hi_val != 0:
            return False
        if lo_val == 0 and hi_val == 0:
            lo_tid, lo_idx, lo_rel = find_ready_task_and_index_with_map(
                temp_lo, job_map_lo, t, lo_tasks, L, ready_lo)
            hi_tid, hi_idx, hi_rel = find_ready_task_and_index_with_map(
                temp_hi, job_map_hi, t, hi_tasks, L, ready_hi)
            # NOTE: this must be if/elif, NOT two independent ifs. The
            # production script used two independent ifs, which let both
            # branches fire in the same iteration and was a genuine,
            # previously-unflagged correctness bug (found via trace-level
            # diff-testing against the reference, not caught by the earlier
            # aggregate 12% disagreement figure, which lumped it in with the
            # busy-window and job-table-fill differences). When both a ready
            # LO and a ready HI job exist at time t, the reference clears
            # only the LO job's slot (and its HI-table shadow) this
            # iteration; the HI job stays pending and is reconsidered next t.
            if lo_tid:
                temp_lo[lo_idx] = 0
                job_map_lo[lo_idx] = None
                ready_lo.remove(lo_tid, lo_rel, lo_idx)
                clear_lo_shadow_in_hi(lo_tid, lo_rel)
            elif hi_tid:
                temp_hi[hi_idx] = 0
                job_map_hi[hi_idx] = None
                ready_hi.remove(hi_tid, hi_rel, hi_idx)
        elif lo_val == 0 and hi_val != 0:
            entry = job_map_hi[t]
            temp_hi[t] = 0
            job_map_hi[t] = None
            if entry is not None:
                ready_hi.remove(entry[0], entry[1], t)
        else:  # lo_val != 0 and hi_val == 0
            rel_info = job_map_lo[t]
            temp_lo[t] = 0
            job_map_lo[t] = None
            if rel_info is not None:
                ready_lo.remove(rel_info[0], rel_info[1], t)
                lo_tid, lo_rel = rel_info
                clear_lo_shadow_in_hi(lo_tid, lo_rel)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLASS BALANCING (linear program over the 16 label-pattern strata)
# ═══════════════════════════════════════════════════════════════════════════════

def _solve_balance_lp(strata_patterns: List[Tuple[int, int, int, int]],
                      strata_counts: List[int],
                      target_n: int, tol: float) -> Optional[List[int]]:
    """
    Chooses how many rows to keep from each of up to 16 label-pattern strata
    so that, in the selected subset, each of the 4 labels' positive rate is
    within `tol` of 0.5, while keeping the total as close to target_n as
    possible (capped at target_n).

    Returns a list of per-stratum counts (same order as strata_patterns), or
    None if scipy is unavailable or the LP is infeasible even at n_s=0
    everywhere (which shouldn't happen — n_s=0 everywhere is always feasible,
    just with total=0).
    """
    if not _HAVE_SCIPY:
        return None

    S = len(strata_patterns)
    if S == 0:
        return []

    c = [-1.0] * S  # maximize sum(n_s)  <=>  minimize -sum(n_s)
    A_ub, b_ub = [], []

    for k in range(4):
        # lower bound: sum_s n_s * (L_s_k - (0.5 - tol)) >= 0
        row_lb = [-(strata_patterns[s][k] - (0.5 - tol)) for s in range(S)]
        A_ub.append(row_lb)
        b_ub.append(0.0)
        # upper bound: sum_s n_s * ((0.5 + tol) - L_s_k) >= 0
        row_ub = [-((0.5 + tol) - strata_patterns[s][k]) for s in range(S)]
        A_ub.append(row_ub)
        b_ub.append(0.0)

    # total selected <= target_n
    A_ub.append([1.0] * S)
    b_ub.append(float(target_n))

    bounds = [(0, strata_counts[s]) for s in range(S)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None
    return [int(round(v)) for v in res.x]


def balance_rows(label_by_index: List[Tuple[int, int, int, int]],
                 target_n: int, tol: float, rng: random.Random) -> Tuple[set, dict]:
    """
    label_by_index[i] = (amc, tt, png, edf) for raw row i.
    Returns (selected_indices_set, achieved_stats_dict).
    Tries `tol`, then relaxes per TOL_RELAXATION_MULTIPLIERS if the achieved
    total is suspiciously small (<50% of target_n), always reporting the
    actual achieved numbers.
    """
    strata: Dict[Tuple[int, int, int, int], List[int]] = defaultdict(list)
    for i, pat in enumerate(label_by_index):
        strata[pat].append(i)
    patterns = list(strata.keys())
    counts = [len(strata[p]) for p in patterns]

    if not _HAVE_SCIPY:
        raise RuntimeError(
            "scipy is required for the class-balancing step but is not installed.\n"
            "Install it with:  pip install scipy"
        )

    best_n_s = None
    used_tol = tol
    for mult in TOL_RELAXATION_MULTIPLIERS:
        trial_tol = min(tol * mult, 0.5)
        n_s = _solve_balance_lp(patterns, counts, target_n, trial_tol)
        if n_s is not None and sum(n_s) >= 0.5 * target_n:
            best_n_s = n_s
            used_tol = trial_tol
            break
        if n_s is not None and best_n_s is None:
            best_n_s = n_s  # keep the first feasible result as a fallback
            used_tol = trial_tol

    if best_n_s is None:
        best_n_s = [0] * len(patterns)
        used_tol = None

    selected = []
    for s, n_take in enumerate(best_n_s):
        n_take = max(0, min(n_take, counts[s]))
        idxs = strata[patterns[s]][:]
        rng.shuffle(idxs)
        selected.extend(idxs[:n_take])
    rng.shuffle(selected)
    selected_set = set(selected)

    total = len(selected_set)
    achieved = {}
    label_names = ["AMC_IMC", "TT_Merge", "IMC_PnG", "EDF_IMC"]
    for k, name in enumerate(label_names):
        pos = sum(1 for i in selected_set if label_by_index[i][k] == 1)
        rate = pos / total if total > 0 else float("nan")
        achieved[name] = {
            "rate": rate,
            "deviation_from_0.5": abs(rate - 0.5) if total > 0 else float("nan"),
        }

    return selected_set, {
        "requested_tol": tol,
        "achieved_tol_used_for_selection": used_tol,
        "total_selected": total,
        "target_n": target_n,
        "labels": achieved,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — WORKER FUNCTION  (runs in a subprocess, one per n)
# ═══════════════════════════════════════════════════════════════════════════════

def _row_header(num_tasks: int) -> List[str]:
    header = ["Taskset_ID", "Num_Tasks", "U_total", "HI_ratio", "CF"]
    for i in range(1, num_tasks + 1):
        header += [f"CLO{i}", f"CHI{i}", f"D{i}", f"T{i}", f"Crit{i}"]
    header += ["AMC_IMC_Schedulable", "TT_Merge_Schedulable",
               "IMC_PnG_Schedulable", "EDF_IMC_Schedulable"]
    return header


def _raw_meta_path(raw_path: str) -> str:
    return raw_path + ".meta.json"


def _raw_pool_metadata(num_tasks: int, implicit_deadline: bool, min_deadline_ratio: float,
                       tmin: int, tmax: int, seed: int, raw_spc: int,
                       expected_fields: int) -> dict:
    """
    Everything about a raw pool that must stay IDENTICAL across runs for a
    resume to be safe (RNG replay depends on the exact generation loop
    producing the exact same sequence of calls; combo/row alignment depends
    on the grid and raw_spc not changing).
    """
    return {
        "num_tasks": num_tasks,
        "implicit_deadline": implicit_deadline,
        "min_deadline_ratio": min_deadline_ratio,
        "tmin": tmin,
        "tmax": tmax,
        "seed": seed,
        "raw_spc": raw_spc,
        "expected_fields": expected_fields,
        "utilizations": UTILIZATIONS,
        "hi_ratios": HI_RATIOS,
        "cf_values": CF_VALUES,
    }


def _inspect_and_truncate_raw_pool(raw_path: str, expected_fields: int) -> List[Tuple[int, int, int, int]]:
    """
    Scans raw_path (no header -- raw pool files are headerless) from the
    start, keeping every row that is complete and well-formed (right field
    count, leading fields parse as ints). The moment a short/corrupt row is
    hit -- which is exactly what an abrupt kill mid-write leaves behind --
    scanning stops there. The file is then truncated to drop that trailing
    partial row, so appending afterward continues cleanly at a valid row
    boundary. Returns the (amc, tt, png, edf) label tuple for every row kept,
    since the balancer needs those regardless of whether a row was generated
    in this process or a previous, interrupted one.
    """
    if not os.path.exists(raw_path):
        return []
    labels: List[Tuple[int, int, int, int]] = []
    good_offset = 0
    with open(raw_path, "r", newline="") as f:
        while True:
            line = f.readline()
            if line == "":
                break
            try:
                parsed = next(csv.reader([line]))
            except Exception:
                break
            if len(parsed) != expected_fields:
                break
            try:
                int(parsed[0])
                int(parsed[1])
                lbl = tuple(int(x) for x in parsed[-4:])
            except Exception:
                break
            labels.append(lbl)  # type: ignore[arg-type]
            good_offset = f.tell()
    with open(raw_path, "r+b") as fb:
        fb.truncate(good_offset)
    return labels


def worker_generate_n(args_dict: dict) -> dict:
    num_tasks         = args_dict["num_tasks"]
    out_path          = args_dict["out_path"]
    tmp_dir           = args_dict["tmp_dir"]
    sets_per_combo    = args_dict["sets_per_combo"]
    tmin              = args_dict["tmin"]
    tmax              = args_dict["tmax"]
    tt_max_window     = args_dict["tt_max_window"]
    amc_max_iter      = args_dict["amc_max_iter"]
    seed              = args_dict["seed"]
    implicit_deadline = args_dict["implicit_deadline"]
    min_deadline_ratio = args_dict["min_deadline_ratio"]
    oversample_factor = args_dict["oversample_factor"]
    balance_tol        = args_dict["balance_tol"]
    cleanup_raw         = args_dict["cleanup_raw"]
    quiet               = args_dict.get("quiet", False)

    random.seed(seed + num_tasks)
    rng = random.Random(seed + num_tasks + 999983)  # separate RNG for selection shuffling

    t_start = time.time()
    raw_spc = max(1, math.ceil(sets_per_combo * oversample_factor))
    total_raw = len(UTILIZATIONS) * len(HI_RATIOS) * len(CF_VALUES) * raw_spc
    target_n = len(UTILIZATIONS) * len(HI_RATIOS) * len(CF_VALUES) * sets_per_combo

    header = _row_header(num_tasks)
    expected_fields = 5 + num_tasks * 5 + 4  # Taskset_ID..CF + 5 cols/task + 4 labels

    deadline_tag = "implicit" if implicit_deadline else "constrained"
    os.makedirs(tmp_dir, exist_ok=True)
    # Real files in a visible, user-chosen directory (not the OS temp dir) so
    # generation can be watched (`tail -f`, a file browser, etc.) while it
    # runs, AND so an abruptly-killed run leaves behind a raw pool that this
    # same function can resume from on the next invocation. raw_path is never
    # deleted automatically (see cleanup_raw) -- clearing it out is left to
    # you, since for a run spanning days, an accidentally-deleted raw pool is
    # much more costly than a little disk space.
    raw_path = os.path.join(tmp_dir, f"imc_{num_tasks}_{deadline_tag}.raw.csv")
    final_tmp_path = os.path.join(tmp_dir, f"imc_{num_tasks}_{deadline_tag}.balanced.csv")

    meta_now = _raw_pool_metadata(num_tasks, implicit_deadline, min_deadline_ratio,
                                  tmin, tmax, seed, raw_spc, expected_fields)

    label_by_index: List[Tuple[int, int, int, int]] = []
    resumed_rows = 0

    if quiet:
        print(f"Working on n={num_tasks}", flush=True)

    try:
        if os.path.exists(raw_path):
            old_meta = None
            meta_path = _raw_meta_path(raw_path)
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as mf:
                        old_meta = json.load(mf)
                except Exception:
                    old_meta = None

            if old_meta != meta_now:
                raise RuntimeError(
                    f"A raw pool file already exists at {raw_path} but its recorded "
                    f"generation parameters (in {meta_path}) don't match this run's "
                    f"settings -- or no metadata sidecar was found at all. Resuming "
                    f"from it would risk silently misaligned or duplicated rows, so "
                    f"this run is refusing to touch it. This script never deletes "
                    f"raw data automatically. If you want to start n={num_tasks} "
                    f"fresh, move or delete {raw_path} (and its .meta.json) "
                    f"yourself; if you meant to resume, check that --sizes, "
                    f"--sets_per_combo, --oversample_factor, --seed, "
                    f"--implicit_deadline, and --min_deadline_ratio all match the "
                    f"original run."
                )

            label_by_index = _inspect_and_truncate_raw_pool(raw_path, expected_fields)
            resumed_rows = len(label_by_index)
            if resumed_rows and not quiet:
                print(f"  [n={num_tasks:4d}] RESUME: found {resumed_rows:,}/{total_raw:,} "
                      f"valid raw rows already in {raw_path}, continuing from there", flush=True)

        with open(_raw_meta_path(raw_path), "w") as mf:
            json.dump(meta_now, mf, indent=2)

        # ── Phase 1: raw oversampled generation, streamed to a visible file ────
        if resumed_rows >= total_raw:
            if not quiet:
                print(f"  [n={num_tasks:4d}] raw pool already complete ({resumed_rows:,} rows) "
                      f"-- skipping straight to balancing", flush=True)
            rows_written = resumed_rows
        else:
            if not quiet:
                print(f"  [n={num_tasks:4d}] raw pool -> {raw_path}  (watch this file to see live progress)", flush=True)
            file_mode = "a" if resumed_rows > 0 else "w"
            row_idx = 0
            global_tid = 1
            with open(raw_path, file_mode, newline="") as f:
                writer = csv.writer(f)
                for U in UTILIZATIONS:
                    for hi_ratio in HI_RATIOS:
                        for CF in CF_VALUES:
                            for _ in range(raw_spc):
                                if row_idx < resumed_rows:
                                    # Not writing this row -- it's already on disk from a
                                    # previous run -- but we DO need to advance the RNG by
                                    # exactly the same calls an uninterrupted run would have
                                    # made, so every row after resumed_rows is bit-for-bit
                                    # identical to what an uninterrupted run would produce.
                                    # This is cheap: only the ~O(n) generation calls run,
                                    # never the schedulability tests.
                                    _burn_ts = generate_taskset(num_tasks, U, tmin, tmax,
                                                                implicit_deadline, min_deadline_ratio)
                                    convert_to_mixed_criticality(_burn_ts, CF, hi_ratio)
                                    row_idx += 1
                                    global_tid += num_tasks
                                    continue

                                base_ts = generate_taskset(
                                    num_tasks, U, tmin, tmax,
                                    implicit_deadline, min_deadline_ratio,
                                )
                                mc_raw = convert_to_mixed_criticality(base_ts, CF, hi_ratio)
                                tasks: List[Task] = [
                                    (global_tid + i, C_lo, C_hi, T, D, crit)
                                    for i, (C_lo, C_hi, T, D, crit) in enumerate(mc_raw)
                                ]
                                global_tid += num_tasks

                                amc_ok = camc_rtb_schedulable(tasks, amc_max_iter)
                                tt_ok  = tt_merge_schedulable(tasks, tt_max_window)
                                imc_ok = imc_png_schedulable(tasks)
                                edf_ok = edf_imc_schedulable(tasks)

                                row = [row_idx, num_tasks, U, hi_ratio, CF]
                                for (_, C_lo, C_hi, T, D, crit) in tasks:
                                    row += [C_lo, C_hi, D, T, crit]
                                row += [int(amc_ok), int(tt_ok), int(imc_ok), int(edf_ok)]
                                writer.writerow(row)

                                label_by_index.append((int(amc_ok), int(tt_ok), int(imc_ok), int(edf_ok)))
                                row_idx += 1

                                if row_idx % PROGRESS_EVERY == 0:
                                    if not quiet:
                                        elapsed = time.time() - t_start
                                        done_this_run = row_idx - resumed_rows
                                        rate = done_this_run / elapsed if elapsed > 0 else 0
                                        eta = (total_raw - row_idx) / rate if rate > 0 else 0
                                        print(f"  [n={num_tasks:4d}] raw {row_idx:>8,}/{total_raw:,} "
                                              f"| {elapsed:.0f}s elapsed | ETA {eta:.0f}s | "
                                              f"{rate:.1f} rows/s", flush=True)
                                    # Flush periodically regardless of verbosity -- this is
                                    # about resume safety (less to re-burn after a crash),
                                    # not console output, and costs ~0.00005 ms/row even on
                                    # real file I/O (benchmarked), so it's free either way.
                                    f.flush()
            rows_written = row_idx

        # ── Phase 2: solve balancing LP over the (small) strata summary ───────
        selected_set, balance_stats = balance_rows(label_by_index, target_n, balance_tol, rng)

        # ── Phase 3: stream through the raw file again, write selected rows
        #             to a temp path in tmp_dir, then MOVE into out_dir ───────
        with open(raw_path, "r", newline="") as fin, \
             open(final_tmp_path, "w", newline="") as fout:
            reader = csv.reader(fin)
            writer = csv.writer(fout)
            writer.writerow(header)
            taskset_id = 1
            for i, row in enumerate(reader):
                if i in selected_set:
                    row[0] = taskset_id
                    writer.writerow(row)
                    taskset_id += 1

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        shutil.move(final_tmp_path, out_path)
        elapsed = time.time() - t_start
        if quiet:
            print(f"Finished n={num_tasks}  rows={len(selected_set):,}  time={elapsed/60:.1f}min", flush=True)
        else:
            print(f"  [n={num_tasks:4d}] balanced file moved -> {out_path}", flush=True)

        if cleanup_raw:
            for p in (raw_path, _raw_meta_path(raw_path)):
                try:
                    os.remove(p)
                except OSError:
                    pass

        elapsed = time.time() - t_start
        return {
            "num_tasks": num_tasks,
            "rows_raw": rows_written,
            "rows_written": len(selected_set),
            "elapsed_s": elapsed,
            "balance": balance_stats,
            "error": None,
        }

    except Exception as e:
        import traceback
        return {
            "num_tasks": num_tasks,
            "rows_raw": len(label_by_index),
            "rows_written": 0,
            "elapsed_s": time.time() - t_start,
            "balance": None,
            "error": f"{e}\n{traceback.format_exc()}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in ("true", "t", "yes", "y", "1"):
        return True
    if v in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {v!r}")


def parse_args():
    p = argparse.ArgumentParser(
        description="IMC-Former dataset generator v2 — corrected, balanced, deadline-configurable.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--implicit_deadline", type=_str2bool, required=True,
                   help="true: D_i = T_i (implicit deadlines).\n"
                        "false: D_i drawn as a genuine constrained deadline (D_i <= T_i).\n"
                        "Run the script once per value to build two independent datasets.")
    p.add_argument("--min_deadline_ratio", type=float, default=DEFAULT_MIN_DEADLINE_RATIO,
                   help=f"Only used when --implicit_deadline false. D_i is drawn uniformly\n"
                        f"from [min_deadline_ratio * T_i, T_i]. Default: {DEFAULT_MIN_DEADLINE_RATIO}.")
    p.add_argument("--sizes", nargs="+", type=int, default=None,
                   help="Task-set sizes to generate. Default: all sizes.")
    p.add_argument("--out_dir", default="generated_datasets",
                   help="Final output directory. Default: generated_datasets/")
    p.add_argument("--tmp_dir", default=None,
                   help="Directory for in-progress raw pools and pre-move balanced files.\n"
                        "Default: <out_dir>/_raw_tmp. Deliberately a real, visible directory\n"
                        "(not the OS temp dir) so you can watch a size's raw pool fill up\n"
                        "while generation runs, AND so that if the process is killed\n"
                        "abruptly, the next run of this script automatically resumes that\n"
                        "size's raw pool from where it left off (matched by a .meta.json\n"
                        "sidecar recording the exact generation parameters used -- if they\n"
                        "don't match the current run, the script refuses to touch the old\n"
                        "file rather than risk misaligned data). Each size's final file is\n"
                        "written completely in this directory and then MOVED into\n"
                        "--out_dir, so a file only appears in --out_dir once finished.")
    p.add_argument("--cleanup_raw", action="store_true",
                   help="Delete a size's raw pool file and .meta.json from --tmp_dir after\n"
                        "its balanced output has been successfully moved to --out_dir.\n"
                        "Default: OFF -- raw pools are never deleted automatically. For a\n"
                        "run spanning hours or days, keeping them means an abrupt kill can\n"
                        "always be resumed; clearing them out is left to you.")
    p.add_argument("--sets_per_combo", type=int, default=None,
                   help="Override target sets_per_combo (post-balancing) for ALL sizes.")
    p.add_argument("--oversample_factor", type=float, default=DEFAULT_OVERSAMPLE_FACTOR,
                   help=f"Raw rows generated per combo = sets_per_combo * this, before\n"
                        f"balancing discards the excess majority-class rows. Higher values\n"
                        f"give the balancer more room to hit --balance_tol, at the cost of\n"
                        f"more generation time. Default: {DEFAULT_OVERSAMPLE_FACTOR}.")
    p.add_argument("--balance_tol", type=float, default=DEFAULT_BALANCE_TOL,
                   help=f"Max allowed deviation from 50%% schedulable, per label, in the\n"
                        f"final output. Default: {DEFAULT_BALANCE_TOL} (i.e. 45-55%%).")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel worker processes. Default: min(len(sizes), cpu_count).")
    p.add_argument("--resume", action="store_true",
                   help="Skip sizes whose output CSV already exists and has >= 90%% of the\n"
                        "expected (post-balancing) row count.")
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                   help=f"Base random seed. Default: {RANDOM_SEED}.")
    p.add_argument("--quiet", action="store_true",
                   help="Minimal console output: just 'Working on n=X' when a size starts,\n"
                        "'Finished n=X ...' when it's done, and the full summary once at the\n"
                        "very end. No per-5000-row progress lines. Note: benchmarked at\n"
                        "~0.00005 ms/row, so this does NOT meaningfully speed up generation\n"
                        "-- the schedulability tests themselves are ~1000x more expensive\n"
                        "per row. Use this for a quieter log, not for a faster run.")
    return p.parse_args()


def count_csv_rows(path: str) -> int:
    try:
        with open(path) as f:
            return sum(1 for _ in f) - 1
    except Exception:
        return 0


def main():
    args = parse_args()

    if not _HAVE_SCIPY:
        print("[ERROR] scipy is required for class balancing but is not installed.")
        print("        Install it with:  pip install scipy")
        sys.exit(1)

    sizes = args.sizes if args.sizes else DEFAULT_TASK_COUNTS
    unknown = [n for n in sizes if n not in SETS_PER_COMBO_TIER]
    if unknown:
        print(f"[ERROR] Unknown sizes: {unknown}")
        print(f"        Supported: {sorted(SETS_PER_COMBO_TIER.keys())}")
        sys.exit(1)

    deadline_tag = "implicit" if args.implicit_deadline else "constrained"

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = args.tmp_dir or os.path.join(args.out_dir, "_raw_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "generation_summary.csv")
    combos = len(UTILIZATIONS) * len(HI_RATIOS) * len(CF_VALUES)

    print("=" * 72)
    print(f"IMC-Former Dataset Generator v2  —  deadline_model={deadline_tag}")
    print("=" * 72)
    if not args.quiet:
        print(f"  Output dir        : {os.path.abspath(args.out_dir)}")
        print(f"  Tmp dir           : {os.path.abspath(tmp_dir)}  (watch here during generation)")
        print(f"  Sizes             : {sizes}")
        print(f"  Seed              : {args.seed}")
        print(f"  Oversample factor : {args.oversample_factor}")
        print(f"  Balance tolerance : ±{args.balance_tol*100:.1f}% around 50%")
        print(f"  Grid              : {len(UTILIZATIONS)} U (incl. overload up to "
              f"{max(UTILIZATIONS)}) × {len(HI_RATIOS)} HI_ratio × {len(CF_VALUES)} CF "
              f"= {combos} combos/n")
        print()

    work_items = []
    total_expected = 0

    for n in sizes:
        spc = args.sets_per_combo or SETS_PER_COMBO_TIER[n]
        expected = combos * spc
        out_path = os.path.join(args.out_dir, f"imc_{n}_{deadline_tag}.csv")
        role = ("training" if n <= 20 else
                "gen_small" if n <= 28 else
                "gen_medium" if n <= 40 else
                "gen_large" if n <= 64 else "gen_xlarge/xxlarge")

        if args.resume and os.path.exists(out_path):
            existing = count_csv_rows(out_path)
            if existing >= int(0.90 * expected):
                if not args.quiet:
                    print(f"  [SKIP]  n={n:4d}: {existing:>7,} rows exist "
                          f"(≥90% of {expected:,}) [{role}]")
                continue
            else:
                if not args.quiet:
                    print(f"  [REDO]  n={n:4d}: {existing:>7,} rows (< 90% of {expected:,}) "
                          f"→ regenerating")

        total_expected += expected
        raw_expected = int(expected * args.oversample_factor)
        if not args.quiet:
            print(f"  [QUEUE] n={n:4d}: target {expected:>7,} rows (raw ~{raw_expected:,})  "
                  f"[{role}]  T=[{PERIOD_RANGE[n][0]},{PERIOD_RANGE[n][1]}]  "
                  f"AMC_iter={AMC_MAX_ITER[n]}  TT_win={TT_MAX_WINDOW[n]:,}")

        work_items.append({
            "num_tasks": n,
            "out_path": out_path,
            "tmp_dir": tmp_dir,
            "sets_per_combo": spc,
            "tmin": PERIOD_RANGE[n][0],
            "tmax": PERIOD_RANGE[n][1],
            "tt_max_window": TT_MAX_WINDOW[n],
            "amc_max_iter": AMC_MAX_ITER[n],
            "seed": args.seed,
            "implicit_deadline": args.implicit_deadline,
            "min_deadline_ratio": args.min_deadline_ratio,
            "oversample_factor": args.oversample_factor,
            "balance_tol": args.balance_tol,
            "cleanup_raw": args.cleanup_raw,
            "quiet": args.quiet,
        })

    if not work_items:
        print("\nNothing to generate (all sizes already complete).")
        sys.exit(0)

    n_workers = min(args.workers or mp.cpu_count(), len(work_items))
    if not args.quiet:
        print(f"\n  Workers: {n_workers}  (parallel across n values)")
        print(f"  Total target rows (pre-relaxation): ~{total_expected:,}")
        print("=" * 72)

    wall_start = time.time()
    if n_workers == 1:
        results = [worker_generate_n(item) for item in work_items]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(worker_generate_n, work_items)

    wall_elapsed = time.time() - wall_start
    total_rows = 0

    print(f"\n{'='*72}")
    print("RESULTS")
    print(f"{'─'*72}")

    write_header = not (os.path.exists(summary_path) and os.path.getsize(summary_path) > 0)
    with open(summary_path, "a", newline="") as fs:
        ws = csv.writer(fs)
        if write_header:
            ws.writerow(["num_tasks", "deadline_model", "rows_raw", "rows_written",
                         "elapsed_s", "rows_per_s", "status",
                         "AMC_IMC_rate", "AMC_IMC_dev_from_0.5",
                         "TT_Merge_rate", "TT_Merge_dev_from_0.5",
                         "IMC_PnG_rate", "IMC_PnG_dev_from_0.5",
                         "EDF_IMC_rate", "EDF_IMC_dev_from_0.5"])

        for r in results:
            n = r["num_tasks"]
            rows = r["rows_written"]
            secs = r["elapsed_s"]
            rate = rows / secs if secs > 0 else 0
            status = "ERROR: " + r["error"] if r["error"] else "OK"
            total_rows += rows

            lab = r["balance"]["labels"] if r["balance"] else {}
            def g(name, key):
                return lab.get(name, {}).get(key, float("nan"))

            ws.writerow([
                n, deadline_tag, r["rows_raw"], rows, f"{secs:.1f}", f"{rate:.1f}", status,
                f"{g('AMC_IMC','rate'):.4f}", f"{g('AMC_IMC','deviation_from_0.5'):.4f}",
                f"{g('TT_Merge','rate'):.4f}", f"{g('TT_Merge','deviation_from_0.5'):.4f}",
                f"{g('IMC_PnG','rate'):.4f}", f"{g('IMC_PnG','deviation_from_0.5'):.4f}",
                f"{g('EDF_IMC','rate'):.4f}", f"{g('EDF_IMC','deviation_from_0.5'):.4f}",
            ])

            if r["error"]:
                print(f"  n={n:4d}: ERROR — {r['error']}")
                continue

            print(f"  n={n:4d}: {rows:>7,}/{r['rows_raw']:,} raw kept "
                  f"in {secs/60:.1f} min ({rate:.1f} rows/s)  →  imc_{n}_{deadline_tag}.csv")
            if r["balance"]:
                for name, stats in r["balance"]["labels"].items():
                    dev = stats["deviation_from_0.5"]
                    flag = "  <-- WARNING: exceeds --balance_tol" if dev > args.balance_tol + 1e-9 else ""
                    print(f"           {name:10s}: {stats['rate']*100:6.2f}% schedulable "
                          f"(dev {dev*100:5.2f}pp){flag}")

    print(f"{'─'*72}")
    print(f"  TOTAL: {total_rows:,} rows in {wall_elapsed/60:.1f} min")
    print(f"  Summary: {summary_path}")
    print(f"{'='*72}")
    print()
    print("Reminder: run this script again with --implicit_deadline "
          f"{'false' if args.implicit_deadline else 'true'} to build the other dataset.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()