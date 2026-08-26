# Coded by Dr. Lalatendu Behera

import random, math, csv, os
from typing import List, Tuple, Dict, Optional

# ----------------------- Tunables -----------------------
RANDOM_SEED      = 42
TMIN, TMAX       = 10, 100
SETS_PER_COMBO   = 1000
UTILIZATIONS     = [round(0.10 + 0.05*i, 2) for i in range(int((0.95 - 0.10) / 0.05) + 1)]
HI_RATIOS        = [round(0.4 + 0.1*i, 2)  for i in range(int((0.8 - 0.4) / 0.1) + 1)]
CF_VALUES        = list(range(2, 7))
TASK_COUNTS      = list(range(4, 21, 4))

WRITE_DETAILED   = True

MAX_WINDOW       = 200_000
FALLBACK_K       = 10
VERBOSE_SELECT   = False
PROGRESS_EVERY   = 100

Task = Tuple[int, int, int, int, int, int]
# (Task_ID, C_lo, C_hi, T, D, Criticality)

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
                     Tmin: int = 10, Tmax: int = 100) -> List[Tuple[int, int, int]]:
    utils   = uunifast(n, U_total)
    periods = generate_periods(n, Tmin, Tmax)
    exec_times = [max(1, int(round(u * T))) for u, T in zip(utils, periods)]
    deadlines  = periods[:]  
    return list(zip(exec_times, periods, deadlines))

def convert_to_mixed_criticality(taskset: List[Tuple[int, int, int]],
                                 CF: int,
                                 hi_ratio: float) -> List[Tuple[int, int, int, int, int]]:
    n = len(taskset)
    n_hi = int(round(n * hi_ratio))
    hi_indices = set(random.sample(range(n), n_hi))
    mc_taskset = []
    for i, (C_lo, T, D) in enumerate(taskset):
        crit = 1 if i in hi_indices else 0
        if crit == 1:
            C_hi = max(1, int(round(C_lo * CF)))
        else:
            cand = max(1, int(round(C_lo / CF)))
            C_hi = 0 if cand == C_lo else cand
        mc_taskset.append((C_lo, C_hi, T, D, crit))
    return mc_taskset

# ----------------------- IMC-PnG ----------------------------

def imc_png_assign_vd(tasks: List[Task]):
    hi_tasks = [t for t in tasks if t[5] == 1]

    U_LO = sum(t[1] / t[3] for t in tasks if t[5] == 0)

    zi = {}
    for t in hi_tasks:
        Clo, Chi, T = t[1], t[2], t[3]
        zi[t[0]] = Clo / T

    slack = 1.0 - (U_LO + sum(zi.values()))
    if slack <= 0:
        return None

    while slack > 1e-9:
        best_tid = None
        best_deriv = float("inf")

        for t in hi_tasks:
            tid, Clo, Chi, T = t[0], t[1], t[2], t[3]
            uL = Clo / T
            uH = Chi / T

            if zi[tid] >= uH:
                continue

            denom = (zi[tid] - uL)**2 + 1e-12
            deriv = ((1 - zi[tid]) * (uH - uL)) / denom

            if deriv < best_deriv:
                best_deriv = deriv
                best_tid = tid

        if best_tid is None:
            break

        increment = min(slack, 0.01)
        zi[best_tid] += increment
        slack -= increment

    xi = {}
    for t in hi_tasks:
        tid, Clo, T = t[0], t[1], t[3]
        xi[tid] = (Clo / T) / zi[tid]

    return xi


def imc_png_schedulable(tasks: List[Task]) -> bool:

    U_LO = sum(t[1] / t[3] for t in tasks if t[5] == 0)
    U_LO_deg = sum(t[2] / t[3] for t in tasks if t[5] == 0)

    HI_tasks = [t for t in tasks if t[5] == 1]
    if not HI_tasks:
        return U_LO <= 1.0 + 1e-12

    uL = []
    uH = []

    for t in HI_tasks:
        Clo, Chi, T = t[1], t[2], t[3]
        uL.append(Clo / T)
        uH.append(Chi / T)

    S = 1.0 - U_LO
    if S <= 0:
        return False

    if sum(uL) > S:
        return False

    k = []
    for i in range(len(uL)):
        if uL[i] == 0:
            k.append(0.0)
        else:
            k.append(math.sqrt((uH[i] - uL[i]) / uL[i]))

    def sum_constraint(lambda_val):
        total = 0.0
        for i in range(len(uL)):
            x_i = 1.0 / (1.0 + k[i] * math.sqrt(lambda_val))
            total += uL[i] / x_i
        return total

    low = 0.0
    high = 1e6

    for _ in range(100):
        mid = (low + high) / 2.0
        if sum_constraint(mid) > S:
            high = mid   
        else:
            low = mid

    lambda_star = (low + high) / 2.0


    x = []
    for i in range(len(uL)):
        x_i = 1.0 / (1.0 + k[i] * math.sqrt(lambda_star))
        x.append(x_i)

    if sum(uL[i] / x[i] for i in range(len(uL))) > S + 1e-12:
        return False

    cond2 = U_LO_deg
    for i in range(len(uL)):
        cond2 += (uH[i] - uL[i]) / (1.0 - x[i] + 1e-12)

    return cond2 <= 1.0 + 1e-12

# ----------------------- EDF-IMC (Global-x IMC) -------------------------
def edf_imc_schedulable(tasks: List[Task]) -> bool:

    U_LO = sum(t[1] / t[3] for t in tasks if t[5] == 0)
    U_LO_deg = sum(t[2] / t[3] for t in tasks if t[5] == 0)

    HI_tasks = [t for t in tasks if t[5] == 1]

    U_HI_LO = sum(t[1] / t[3] for t in HI_tasks)
    U_HI_HI = sum(t[2] / t[3] for t in HI_tasks)

    denom = 1.0 - U_LO
    if denom <= 0:
        return False

    if U_HI_LO > denom:
        return False

    x = U_HI_LO / denom

    if U_LO + (U_HI_LO / x) > 1.0 + 1e-12:
        return False

    cond2 = U_LO_deg
    for t in HI_tasks:
        Clo, Chi, T = t[1], t[2], t[3]
        uL = Clo / T
        uH = Chi / T
        cond2 += (uH - uL) / (1.0 - x + 1e-12)

    return cond2 <= 1.0 + 1e-12

# ------------------- AMC-IMC Test -------------------------

def compute_R_lo(task_i, hp_tasks, max_iter=1000):
    _, C_lo_i, _, _, _, _ = task_i
    R = C_lo_i

    for _ in range(max_iter):
        interference = 0
        for task_j in hp_tasks:
            _, C_lo_j, _, T_j, _, _ = task_j
            interference += math.ceil(R / T_j) * C_lo_j

        R_next = C_lo_i + interference

        if R_next == R:
            return R
        R = R_next

    return float('inf')


def compute_R_hi(task_i, hp_tasks, hpL_tasks, R_lo_i, max_iter=1000):
    _, C_lo_i, C_hi_i, _, _, _ = task_i

    C_i = max(C_lo_i, C_hi_i)
    R = C_i

    for _ in range(max_iter):
        interference_hi = 0
        for task_j in hp_tasks:
            _, _, C_hi_j, T_j, _, _ = task_j
            interference_hi += math.ceil(R / T_j) * C_hi_j

        interference_lo = 0
        for task_j in hpL_tasks:
            _, C_lo_j, C_hi_j, T_j, _, _ = task_j
            interference_lo += math.ceil(R_lo_i / T_j) * (C_lo_j - C_hi_j)

        R_next = C_i + interference_hi + interference_lo

        if R_next == R:
            return R
        R = R_next

    return float('inf')


def camc_rtb_schedulable(tasks: List[Task]) -> bool:

    tasks_sorted = sorted(tasks, key=lambda x: x[3])

    R_lo_list = []

    for i, task_i in enumerate(tasks_sorted):
        hp_tasks = tasks_sorted[:i]

        R_lo = compute_R_lo(task_i, hp_tasks)
        D_i = task_i[4]

        if R_lo > D_i:
            return False

        R_lo_list.append(R_lo)

    for i, task_i in enumerate(tasks_sorted):
        hp_tasks = tasks_sorted[:i]
        hpL_tasks = [t for t in hp_tasks if t[5] == 0]

        R_hi = compute_R_hi(task_i, hp_tasks, hpL_tasks, R_lo_list[i])
        D_i = task_i[4]

        if R_hi > D_i:
            return False

    return True

# ----------------------- EDF-VD -------------------------
def edf_vd_schedulable(tasks: List[Task]) -> bool:
    U_LO_LO = sum(Clo / T for (_tid, Clo, _Chi, T, _D, crit) in tasks if crit == 0)
    U_HI_LO = sum(Clo / T for (_tid, Clo, _Chi, T, _D, crit) in tasks if crit == 1)
    U_HI_HI = sum(Chi / T for (_tid, _Clo, Chi, T, _D, crit) in tasks if crit == 1)
    denom = 1.0 - U_LO_LO
    if denom <= 0: return False
    if U_HI_LO > denom: return False
    x = U_HI_LO / denom
    return (x * U_LO_LO + U_HI_HI) <= 1.0 + 1e-12

# ----------------------- TT-Merge helpers ----------------
def edf_busy_window(tasks: List[Task], cap: int) -> int:
    if not tasks: return 0
    def Cb(t: Task) -> int: return max(t[1], t[2])
    W = max(sum(Cb(t) for t in tasks), 1)
    while True:
        W_next = 0
        for (_tid, Clo, Chi, T, _D, _crit) in tasks:
            W_next += math.ceil(W / T) * max(Clo, Chi)
            if W_next > cap: return cap + 1
        if W_next == W: return min(W_next, cap)
        if W_next > cap: return cap + 1
        W = W_next

def generate_jobs_table_and_map(tasks: List[Task], L: int, use_hi_exec: bool):
    jobs = []
    for (tid, Clo, Chi, T, D, _crit) in tasks:
        C = Chi if use_hi_exec else Clo
        if C <= 0: continue
        for r in range(0, L, T):
            dl = min(r + D, L)
            jobs.append({"tid": tid, "release": r, "deadline": dl, "C": C, "T": T, "D": D})
    jobs.sort(key=lambda j: (j["deadline"], j["release"], j["tid"]))
    table = [0]*L
    job_map: List[Optional[Tuple[int,int]]] = [None]*L
    for j in reversed(jobs):
        rem = j["C"]; t = j["deadline"] - 1
        while rem > 0 and t >= j["release"]:
            if t < 0: break
            if table[t] == 0:
                table[t] = j["tid"]; job_map[t] = (j["tid"], j["release"]); rem -= 1
            t -= 1
    return table, job_map

def find_ready_task_and_index_with_map(temp_table, job_map, t, tasks: List[Task], L: int):
    best = None  
    for (tid, _Clo, _Chi, T, D, _crit) in tasks:
        rel = (t // T) * T
        start, end = rel, min(rel + D, L)
        if not (start <= t < end): continue
        idx = None
        for i in range(start, end):
            if job_map[i] == (tid, rel): idx = i; break
        if idx is not None:
            deadline = end
            if best is None or (deadline < best[1]) or (deadline == best[1] and tid < best[0]):
                best = (tid, deadline, idx, rel)
    if best: return best[0], best[2], best[3]
    return None, None, None

def tt_merge_schedulable(tasks: List[Task]) -> bool:
    W = edf_busy_window(tasks, MAX_WINDOW)
    if W > MAX_WINDOW:
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
                    start, end = rel, min(rel + D, L)
                    cnt = diff
                    for i in range(end-1, start-1, -1):
                        if temp_hi[i] == tid:
                            temp_hi[i] = 0; job_map_hi[i] = None; cnt -= 1
                            if cnt == 0: break

    D_of_lo: Dict[int,int] = {tid: D for (tid, _Clo, _Chi, _T, D, _crit) in lo_tasks}

    for t in range(L):
        lo_val = temp_lo[t]; hi_val = temp_hi[t]
        if lo_val != 0 and hi_val != 0: return False
        if lo_val == 0 and hi_val == 0:
            lo_tid, lo_idx, lo_rel = find_ready_task_and_index_with_map(temp_lo, job_map_lo, t, lo_tasks, L)
            hi_tid, hi_idx, hi_rel = find_ready_task_and_index_with_map(temp_hi, job_map_hi, t, hi_tasks, L)
            if lo_tid and hi_tid:
                temp_lo[lo_idx] = 0; job_map_lo[lo_idx] = None
                D_lo = D_of_lo.get(lo_tid, 0)
                for i in range(lo_rel, min(lo_rel + D_lo, L)):
                    if job_map_hi[i] == (lo_tid, lo_rel): temp_hi[i] = 0; job_map_hi[i] = None
            elif lo_tid:
                temp_lo[lo_idx] = 0; job_map_lo[lo_idx] = None
                D_lo = D_of_lo.get(lo_tid, 0)
                for i in range(lo_rel, min(lo_rel + D_lo, L)):
                    if job_map_hi[i] == (lo_tid, lo_rel): temp_hi[i] = 0; job_map_hi[i] = None
            elif hi_tid:
                temp_hi[hi_idx] = 0; job_map_hi[hi_idx] = None
        elif lo_val == 0 and hi_val != 0:
            temp_hi[t] = 0; job_map_hi[t] = None
        elif lo_val != 0 and hi_val == 0:
            rel_info = job_map_lo[t]; temp_lo[t] = 0; job_map_lo[t] = None
            if rel_info:
                lo_tid, lo_rel = rel_info
                D_lo = D_of_lo.get(lo_tid, 0)
                for i in range(lo_rel, min(lo_rel + D_lo, L)):
                    if job_map_hi[i] == (lo_tid, lo_rel): temp_hi[i] = 0; job_map_hi[i] = None
    return True

# ----------------------- Main ---------------------------
if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    detailed_csv = os.path.join(script_dir, "General_task_set_generated.csv")
    summary_csv  = os.path.join(script_dir, "General_Results_summary.csv")

    processed_global = 0
    taskset_id = 1
    global_task_id = 1

    fs = open(summary_csv, "w", newline="")
    ws = csv.writer(fs)
    ws.writerow(["Num_Tasks", "U_total", "HI_ratio", "CF",
                 "Total_Generated",
                 "AMC_IMC_Schedulable_Count",
                 "TTA_Schedulable_Count",
                 "IMC_PnG_Schedulable_Count",
                 "EDF_IMC_Schedulable_Count",
                 "AMC_IMC_Schedulable_accpt",
                 "TTA_Schedulable_accpt",
                 "IMC_PnG_Schedulable_accpt",
                 "EDF_IMC_Schedulable_accpt"])
    fs.flush(); os.fsync(fs.fileno())

    detailed_files = {}
    detailed_writers = {}

    if WRITE_DETAILED:
        for n in TASK_COUNTS:
            filename = os.path.join(script_dir, f"General_task_set_{n}.csv")
            f = open(filename, "w", newline="")
            w = csv.writer(f)

            header = ["Taskset_ID", "Num_Tasks", "U_total", "HI_ratio", "CF"]
            for i in range(1, n + 1):
                header.extend([
                    f"CLO{i}", f"CHI{i}", f"D{i}", f"T{i}", f"Crit{i}"
                ])
            header.extend([
                "AMC_IMC_Schedulable",
                "TT_Merge_Schedulable",
                "IMC_PnG_Schedulable",
                "EDF_IMC_Schedulable"
            ])
            w.writerow(header)
            f.flush()
            os.fsync(f.fileno())
            detailed_files[n]=f
            detailed_writers[n]=w
    else:
        detailed_files={}
        detailed_writers={}

    try:
        for num_tasks in TASK_COUNTS:
            for U in UTILIZATIONS:
                for hi_ratio in HI_RATIOS:
                    for CF in CF_VALUES:

                        amc_count      = 0
                        tt_count       = 0
                        imc_count      = 0
                        edf_imc_count  = 0   
                        amc_accpt      = 0
                        tt_accpt       = 0
                        imc_accpt      = 0
                        edf_imc_accpt  = 0

                        for _ in range(SETS_PER_COMBO):
                            base_ts   = generate_taskset(num_tasks, U, TMIN, TMAX)
                            mc_ts_raw = convert_to_mixed_criticality(base_ts, CF, hi_ratio)

                            tasks: List[Task] = []
                            for (C_lo, C_hi, T, D, crit) in mc_ts_raw:
                                tasks.append((global_task_id, C_lo, C_hi, T, D, crit))
                                global_task_id += 1

                            amc_ok      = camc_rtb_schedulable(tasks)
                            tt_ok       = tt_merge_schedulable(tasks)
                            imc_ok      = imc_png_schedulable(tasks)
                            edf_imc_ok  = edf_imc_schedulable(tasks)  

                            if amc_ok:     amc_count += 1
                            if tt_ok:      tt_count += 1
                            if imc_ok:     imc_count += 1
                            if edf_imc_ok: edf_imc_count += 1

                            if WRITE_DETAILED:
                                row = [taskset_id, num_tasks, U, hi_ratio, CF]

                                for (_, C_lo, C_hi, T, D, crit) in tasks:
                                    row.extend([C_lo, C_hi, D, T, crit])

                                row.extend([
                                    int(amc_ok),
                                    int(tt_ok),
                                    int(imc_ok),
                                    int(edf_imc_ok)
                                ])

                                detailed_writers[num_tasks].writerow(row)
                                if taskset_id % PROGRESS_EVERY == 0:
                                    print(f"Processed {taskset_id} task sets...")
                                    detailed_files[num_tasks].flush()
                                    os.fsync(detailed_files[num_tasks].fileno())

                            taskset_id += 1
                            processed_global += 1

                            if not WRITE_DETAILED and processed_global % PROGRESS_EVERY == 0:
                                print(f"Processed {processed_global} task sets...")


                        amc_accpt = amc_count/SETS_PER_COMBO
                        tt_accpt = tt_count/SETS_PER_COMBO
                        imc_accpt = imc_count/SETS_PER_COMBO
                        edf_imc_accpt = edf_imc_count/SETS_PER_COMBO

                        ws.writerow([num_tasks, U, hi_ratio, CF,
                                     SETS_PER_COMBO,
                                     amc_count,
                                     tt_count,
                                     imc_count,
                                     edf_imc_count,
                                     amc_accpt,
                                     tt_accpt,
                                     imc_accpt,
                                     edf_imc_accpt])
                        fs.flush()
                        os.fsync(fs.fileno())

                        print(f"Summary updated for combo: "
                              f"n={num_tasks}, U={U}, HI_ratio={hi_ratio}, CF={CF} | "
                              f"AMC_IMC={amc_count}/{SETS_PER_COMBO}, "
                              f"TTA={tt_count}/{SETS_PER_COMBO}, "
                              f"IMC={imc_count}/{SETS_PER_COMBO}, "
                              f"EDF-IMC={edf_imc_count}/{SETS_PER_COMBO}, "
                              f"AMC_IMC_ratio={amc_accpt}/{SETS_PER_COMBO}, "
                              f"TTA_ratio={tt_accpt}/{SETS_PER_COMBO}, "
                              f"IMC_ratio={imc_accpt}/{SETS_PER_COMBO}, "
                              f"EDF-IMC_ratio={edf_imc_accpt}/{SETS_PER_COMBO}")

    finally:
        fs.close()
        for f in detailed_files.values():
            f.close()

    print(f"✅ Done.\n   Summary  -> {summary_csv}\n   Detailed -> {detailed_csv if WRITE_DETAILED else '(skipped)'}")


