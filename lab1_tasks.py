"""
Lab 1: Fork-Join Model, Team Creation, Thread Scoping (Python edition)
Tasks 1.1, 1.2, 1.3

Usage:
    pip install numba psutil        # numba и psutil желательны, но необязательны
    python lab1_tasks.py            # запускает все три таска
    python lab1_tasks.py 1.1        # только один таск (1.1 / 1.2 / 1.3)

Результаты пишутся в папку ./results:
    task1_1_output.txt   - stdout 10 запусков
    task1_2_oversub.csv  - время создания/join команды потоков
    task1_3_saturation.csv - время и загрузка CPU при тяжёлой нагрузке
"""
import contextlib
import csv
import math
import os
import platform
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

try:
    from numba import njit
except ImportError:
    njit = None
try:
    import psutil
except ImportError:
    psutil = None

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Starter code (fork-join emulation)
# ----------------------------------------------------------------------------
def worker_task(thread_id: int, team_size: int):
    native_tid = threading.get_native_id()
    role = "Master" if thread_id == 0 else "Worker"
    time.sleep(0.001 * (thread_id % 3))
    print(f"[{role}] Logical Rank: {thread_id} of {team_size} | Native OS TID: {native_tid}")


def run_team(num_threads: int):
    print(f"--- Forking a team of {num_threads} threads ---")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_task, tid, num_threads) for tid in range(num_threads)]
        for f in futures:
            f.result()  # implicit barrier
    print("--- Joined thread team. Execution returned to serial master ---\n")


# ----------------------------------------------------------------------------
# Task 1.1: non-determinism, 10 consecutive runs -> text file
# ----------------------------------------------------------------------------
def task_1_1(runs: int = 10, team: int = 4):
    path = os.path.join(RESULTS_DIR, "task1_1_output.txt")
    with open(path, "w", encoding="utf-8") as fh, contextlib.redirect_stdout(fh):
        for r in range(1, runs + 1):
            print(f"===== RUN {r} =====")
            run_team(team)
    print(f"[Task 1.1] {runs} runs saved to {path}")
    print("           Сравните порядок строк между запусками (RUN 1..10).")


# ----------------------------------------------------------------------------
# Task 1.2: oversubscription sweep P in {1,2,4,8,16,32,64}
# ----------------------------------------------------------------------------
def _noop(_tid):
    return threading.get_native_id()


def time_team_creation(p: int) -> float:
    """Wall-clock time to create the team, run a trivial task, and join."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=p) as ex:
        # small sleep forces all P workers to actually be spawned
        futs = [ex.submit(lambda i=i: (time.sleep(0.001), _noop(i))) for i in range(p)]
        for f in futs:
            f.result()
    return time.perf_counter() - t0


def task_1_2(sizes=(1, 2, 4, 8, 16, 32, 64), trials: int = 10):
    rows = []
    time_team_creation(2)  # warm-up
    print("[Task 1.2] Oversubscription sweep (create + join)")
    print(f"{'P':>4} | {'mean, ms':>10} | {'stdev, ms':>10} | {'min, ms':>9}")
    for p in sizes:
        ts = [time_team_creation(p) * 1000 for _ in range(trials)]
        mean, sd, mn = statistics.mean(ts), statistics.pstdev(ts), min(ts)
        rows.append((p, mean, sd, mn))
        print(f"{p:>4} | {mean:>10.3f} | {sd:>10.3f} | {mn:>9.3f}")
    path = os.path.join(RESULTS_DIR, "task1_2_oversub.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["threads", "mean_ms", "stdev_ms", "min_ms", "trials"])
        for p, m, s, mn in rows:
            w.writerow([p, f"{m:.4f}", f"{s:.4f}", f"{mn:.4f}", trials])
    print(f"           saved to {path}\n")


# ----------------------------------------------------------------------------
# Task 1.3: CPU saturation, 10,000,000 sqrt per thread
# ----------------------------------------------------------------------------
N_SQRT = 10_000_000


def _sqrt_loop(n):
    s = 0.0
    for i in range(n):
        s += math.sqrt(i)
    return s


if njit is not None:
    # nogil=True освобождает GIL, поэтому потоки реально работают параллельно
    sqrt_loop = njit(nogil=True)(_sqrt_loop)
    sqrt_loop(10)  # JIT warm-up
    BACKEND = "numba (nogil)"
else:
    sqrt_loop = _sqrt_loop
    BACKEND = "pure Python (GIL: потоки НЕ масштабируются!)"


def _sample_cpu(stop: threading.Event, samples: list):
    psutil.cpu_percent(percpu=True)  # первый вызов сбрасывает счётчик
    while not stop.is_set():
        samples.append(psutil.cpu_percent(interval=0.25, percpu=True))


def run_heavy(p: int):
    samples = []
    stop = threading.Event()
    mon = None
    if psutil is not None:
        mon = threading.Thread(target=_sample_cpu, args=(stop, samples), daemon=True)
        mon.start()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=p) as ex:
        futs = [ex.submit(sqrt_loop, N_SQRT) for _ in range(p)]
        for f in futs:
            f.result()
    elapsed = time.perf_counter() - t0
    if mon:
        stop.set()
        mon.join()
    avg_total = peak_cores = None
    if samples:
        avg_total = statistics.mean(statistics.mean(s) for s in samples)
        # сколько ядер в пике загружено > 50%
        peak_cores = max(sum(1 for c in s if c > 50) for s in samples)
    return elapsed, avg_total, peak_cores


def task_1_3(sizes=(1, 2, 4, 8, 16, 32, 64)):
    print(f"[Task 1.3] CPU saturation | backend: {BACKEND}")
    if psutil is None:
        print("           psutil не установлен -> загрузку CPU смотрите вручную "
              "(htop / Task Manager) и делайте скриншот на время прогона.")
    print(f"{'P':>4} | {'time, s':>8} | {'avg CPU %':>9} | {'busy cores':>10}")
    rows = []
    for p in sizes:
        el, cpu, cores = run_heavy(p)
        rows.append((p, el, cpu, cores))
        cpu_s = f"{cpu:9.1f}" if cpu is not None else "      n/a"
        cor_s = f"{cores:10d}" if cores is not None else "       n/a"
        print(f"{p:>4} | {el:8.3f} | {cpu_s} | {cor_s}")
    path = os.path.join(RESULTS_DIR, "task1_3_saturation.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["threads", "time_s", "avg_cpu_pct", "busy_cores", "backend"])
        for p, el, cpu, cores in rows:
            w.writerow([p, f"{el:.4f}", "" if cpu is None else f"{cpu:.2f}",
                        "" if cores is None else cores, BACKEND])
    print(f"           saved to {path}\n")


# ----------------------------------------------------------------------------
def print_system_info():
    print("=== System info (для Section I отчёта) ===")
    print("OS      :", platform.platform())
    print("CPU     :", platform.processor() or "см. lscpu / Диспетчер задач")
    print("Python  :", sys.version.split()[0])
    print("Logical :", os.cpu_count())
    if psutil:
        print("Physical:", psutil.cpu_count(logical=False))
    print()


if __name__ == "__main__":
    print_system_info()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1.1"):
        task_1_1()
    if which in ("all", "1.2"):
        task_1_2()
    if which in ("all", "1.3"):
        task_1_3()
