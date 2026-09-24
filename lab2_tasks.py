"""
Lab 2: Numerical Integration (Pi) & Parallel Reductions (Python edition)
Tasks 2.1 - 2.4

Usage:
    pip install numpy numba matplotlib
    python lab2_tasks.py            # все таски
    python lab2_tasks.py 2.1        # только один (2.1 / 2.2 / 2.3 / 2.4)
    python lab2_tasks.py all --quick   # уменьшенные N для быстрой проверки

Результаты (CSV + графики) -> ./results
"""
import os

# NUMBA_NUM_THREADS нужно задать ДО импорта numba, чтобы потом менять P до 16
os.environ.setdefault("NUMBA_NUM_THREADS", str(max(16, os.cpu_count() or 1)))

import csv
import statistics
import sys
import threading
import time

import numpy as np
import numba
from numba import njit, prange

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

QUICK = "--quick" in sys.argv
N_STEPS = 10_000_000 if QUICK else 100_000_000      # Tasks 2.1, 2.3
N_CRIT = 100_000 if QUICK else 1_000_000            # Task 2.2 (по заданию 1e6)
TRIALS = 5


# ----------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------
@njit
def pi_serial(num_steps):
    step = 1.0 / num_steps
    total = 0.0
    for i in range(num_steps):
        x = (i + 0.5) * step
        total += 4.0 / (1.0 + x * x)
    return total * step


@njit(parallel=True)
def pi_reduction(num_steps):
    """prange + += -> Numba сам делает private accumulators + reduction."""
    step = 1.0 / num_steps
    total = 0.0
    for i in prange(num_steps):
        x = (i + 0.5) * step
        total += 4.0 / (1.0 + x * x)
    return total * step


@njit(nogil=True)
def pi_race_chunk(shared, start, end, step):
    """
    Variant A: все потоки пишут в ОДНУ общую ячейку shared[0] без синхронизации.
    nogil=True -> потоки реально выполняются параллельно (GIL отпущен).
    Примечание: компилятор (LLVM) может держать shared[0] в регистре внутри цикла
    и записать его один раз в конце. Тогда 'lost updates' происходят на уровне
    финального store (побеждает последний писавший поток), результат всё равно
    неверный и ошибка растёт с P. Это стоит упомянуть в отчёте.
    """
    for i in range(start, end):
        x = (i + 0.5) * step
        shared[0] += 4.0 / (1.0 + x * x)


def pi_naive_race(num_steps, threads):
    shared = np.zeros(1, dtype=np.float64)
    step = 1.0 / num_steps
    chunk = num_steps // threads
    ths = []
    for t in range(threads):
        s = t * chunk
        e = num_steps if t == threads - 1 else s + chunk
        ths.append(threading.Thread(target=pi_race_chunk, args=(shared, s, e, step)))
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    return shared[0] * step


# --- pure-Python versions for Task 2.2 (locks нельзя использовать в nopython) ---
def pi_serial_py(num_steps):
    step = 1.0 / num_steps
    total = 0.0
    for i in range(num_steps):
        x = (i + 0.5) * step
        total += 4.0 / (1.0 + x * x)
    return total * step


def pi_critical(num_steps, threads):
    """Variant B: lock на каждом шаге (эмуляция #pragma omp critical)."""
    lock = threading.Lock()
    shared = [0.0]
    step = 1.0 / num_steps
    chunk = num_steps // threads

    def work(s, e):
        for i in range(s, e):
            x = (i + 0.5) * step
            term = 4.0 / (1.0 + x * x)
            with lock:
                shared[0] += term

    ths = []
    for t in range(threads):
        s = t * chunk
        e = num_steps if t == threads - 1 else s + chunk
        ths.append(threading.Thread(target=work, args=(s, e)))
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    return shared[0] * step


def timeit(fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    return r, time.perf_counter() - t0


def save_csv(name, header, rows):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"   saved -> {path}")


# ----------------------------------------------------------------------------
# Task 2.1: race condition quantification, P = 1,2,4,8
# ----------------------------------------------------------------------------
def task_2_1():
    print(f"\n[Task 2.1] Naive race, N = {N_STEPS:,}")
    pi_race_chunk(np.zeros(1), 0, 10, 0.1)  # warm-up
    print(f"{'P':>3} | {'Pi computed':>16} | {'abs error':>10} | {'time, s':>8}")
    rows = []
    for p in (1, 2, 4, 8):
        val, t = timeit(pi_naive_race, N_STEPS, p)
        err = abs(val - np.pi)
        rows.append((p, f"{val:.12f}", f"{err:.3e}", f"{t:.4f}"))
        print(f"{p:>3} | {val:16.12f} | {err:10.2e} | {t:8.3f}")
    save_csv("task2_1_race.csv", ["threads", "pi", "abs_error", "time_s"], rows)


# ----------------------------------------------------------------------------
# Task 2.2: critical section overhead, N = 1,000,000
# ----------------------------------------------------------------------------
def task_2_2():
    print(f"\n[Task 2.2] Critical section overhead, N = {N_CRIT:,}")
    pi_serial(1000)  # warm-up
    _, t_py = timeit(pi_serial_py, N_CRIT)
    _, t_nb = timeit(pi_serial, N_CRIT)
    print(f"   serial pure Python : {t_py:.4f} s   (основной baseline для overhead)")
    print(f"   serial Numba       : {t_nb:.4f} s   (для справки)")
    print(f"{'P':>3} | {'Pi':>14} | {'time, s':>8} | {'overhead vs serial-Py, %':>24}")
    rows = []
    for p in (1, 2, 4, 8):
        val, t = timeit(pi_critical, N_CRIT, p)
        ovh = (t - t_py) / t_py * 100
        rows.append((p, f"{val:.12f}", f"{t:.4f}", f"{ovh:.1f}", f"{t_py:.4f}", f"{t_nb:.6f}"))
        print(f"{p:>3} | {val:14.10f} | {t:8.3f} | {ovh:24.1f}")
    save_csv("task2_2_critical.csv",
             ["threads", "pi", "time_s", "overhead_pct", "serial_py_s", "serial_numba_s"], rows)


# ----------------------------------------------------------------------------
# Task 2.3 + 2.4: strong scaling, speedup, efficiency, graph
# ----------------------------------------------------------------------------
def task_2_3_and_2_4():
    print(f"\n[Task 2.3/2.4] Strong scaling of reduction, N = {N_STEPS:,}, {TRIALS} trials")
    pi_reduction(1000)  # warm-up (JIT)
    procs = [1, 2, 4, 8, 16]
    max_t = numba.config.NUMBA_NUM_THREADS
    means, rows_raw = {}, []
    for p in procs:
        if p > max_t:
            print(f"   P={p} пропущен: NUMBA_NUM_THREADS={max_t}")
            continue
        numba.set_num_threads(p)
        pi_reduction(1000)
        ts = []
        for k in range(TRIALS):
            val, t = timeit(pi_reduction, N_STEPS)
            ts.append(t)
            rows_raw.append((p, k + 1, f"{t:.6f}", f"{val:.12f}"))
        means[p] = statistics.mean(ts)
    save_csv("task2_3_raw.csv", ["threads", "trial", "time_s", "pi"], rows_raw)

    t1 = means[1]
    rows = []
    print(f"{'P':>3} | {'T(P), s':>9} | {'S(P)':>6} | {'E(P)':>6}")
    for p, tp in means.items():
        s = t1 / tp
        e = s / p
        rows.append((p, f"{tp:.6f}", f"{s:.4f}", f"{e:.4f}"))
        print(f"{p:>3} | {tp:9.4f} | {s:6.2f} | {e:6.2f}")
    save_csv("task2_4_speedup.csv", ["threads", "mean_time_s", "speedup", "efficiency"], rows)

    # --- graph ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   matplotlib не установлен -> график пропущен")
        return
    ps = list(means.keys())
    sp = [t1 / means[p] for p in ps]
    ef = [s / p for s, p in zip(sp, ps)]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(ps, sp, "o-", label="Measured speedup S(P)")
    ax1.plot(ps, ps, "k--", label="Ideal linear speedup")
    ax1.set_xlabel("Threads P")
    ax1.set_ylabel("Speedup")
    ax1.set_xticks(ps)
    ax2 = ax1.twinx()
    ax2.plot(ps, ef, "s:", color="tab:red", label="Efficiency E(P)")
    ax2.set_ylabel("Parallel efficiency")
    ax2.set_ylim(0, 1.15)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    ax1.set_title(f"Pi reduction: speedup & efficiency (N={N_STEPS:,})")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "task2_4_speedup.png")
    fig.savefig(out, dpi=200)
    print(f"   graph saved -> {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = args[0] if args else "all"
    print(f"Logical CPUs: {os.cpu_count()} | Numba threads: {numba.config.NUMBA_NUM_THREADS}")
    if which in ("all", "2.1"):
        task_2_1()
    if which in ("all", "2.2"):
        task_2_2()
    if which in ("all", "2.3", "2.4"):
        task_2_3_and_2_4()
