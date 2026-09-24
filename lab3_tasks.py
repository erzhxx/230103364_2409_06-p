"""
Lab 3: Work-Sharing & Loop Scheduling (Mandelbrot) - Python edition
Tasks 3.1 - 3.4

Реализованы политики, как в OpenMP:
    static            - блоки по N/P строк (schedule(static))
    static,C          - round-robin чанки по C строк (schedule(static, C))
    dynamic,C         - общая очередь чанков, поток берёт следующий, когда свободен
    guided,C          - размер чанка = max(C, remaining / P), уменьшается по ходу

Потоки - обычные threading.Thread, а тяжёлое ядро Mandelbrot - numba nogil,
поэтому GIL не мешает и потоки реально работают параллельно.

Usage:
    pip install numpy numba matplotlib
    python lab3_tasks.py            # полный sweep 4x4, 3 прогона, все политики
    python lab3_tasks.py --quick    # маленькое изображение для проверки

Результаты -> ./results (CSV + heatmap PNG + пример картинки Mandelbrot)
"""
import csv
import os
import statistics
import sys
import threading
import time

import numpy as np
from numba import njit

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

QUICK = "--quick" in sys.argv
WIDTH, HEIGHT = (480, 270) if QUICK else (1920, 1080)
MAX_ITER = 200 if QUICK else 1000
THREADS = [2, 4, 8, 16]
CHUNKS = [1, 16, 64, 256]
RUNS = 3
POLICIES = ["static", "dynamic", "guided"]


# ----------------------------------------------------------------------------
# Kernel: рендерит строки [row_start, row_end), возвращает суммарное число
# итераций (это и есть "работа" - метрика для load imbalance)
# ----------------------------------------------------------------------------
@njit(nogil=True)
def render_rows(img, row_start, row_end, width, height, max_iter):
    total_iters = 0
    for py in range(row_start, row_end):
        y0 = (py - height / 2.0) * 4.0 / height
        for px in range(width):
            x0 = (px - width / 2.0) * 4.0 / width
            x = 0.0
            y = 0.0
            it = 0
            while x * x + y * y <= 4.0 and it < max_iter:
                xt = x * x - y * y + x0
                y = 2.0 * x * y + y0
                x = xt
                it += 1
            img[py, px] = it
            total_iters += it
    return total_iters


# ----------------------------------------------------------------------------
# Task 3.1: schedulers
# ----------------------------------------------------------------------------
def run_schedule(policy, P, C, img):
    """
    Возвращает (elapsed_s, work_per_thread[list], rows_per_thread[list]).
    policy: 'static' | 'dynamic' | 'guided'
    C: chunk size. Для static берём schedule(static, C) - round-robin по C строк.
       (C=None -> классический static: P непрерывных блоков)
    """
    work = [0] * P
    rows = [0] * P
    lock = threading.Lock()
    next_row = [0]

    def claim_dynamic():
        with lock:
            s = next_row[0]
            if s >= HEIGHT:
                return None
            e = min(s + C, HEIGHT)
            next_row[0] = e
            return s, e

    def claim_guided():
        with lock:
            s = next_row[0]
            if s >= HEIGHT:
                return None
            remaining = HEIGHT - s
            size = max(C, remaining // P)
            e = min(s + size, HEIGHT)
            next_row[0] = e
            return s, e

    def worker(tid):
        w = r = 0
        if policy == "static":
            if C is None:
                blk = (HEIGHT + P - 1) // P
                spans = [(tid * blk, min((tid + 1) * blk, HEIGHT))]
            else:
                spans = []
                s = tid * C
                while s < HEIGHT:
                    spans.append((s, min(s + C, HEIGHT)))
                    s += P * C
            for s, e in spans:
                if s < e:
                    w += render_rows(img, s, e, WIDTH, HEIGHT, MAX_ITER)
                    r += e - s
        else:
            claim = claim_dynamic if policy == "dynamic" else claim_guided
            while True:
                span = claim()
                if span is None:
                    break
                s, e = span
                w += render_rows(img, s, e, WIDTH, HEIGHT, MAX_ITER)
                r += e - s
        work[tid], rows[tid] = w, r

    ths = [threading.Thread(target=worker, args=(t,)) for t in range(P)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return time.perf_counter() - t0, work, rows


def imbalance(work):
    """Imbalance = (Max - Min) / Average (Task 3.4)."""
    avg = sum(work) / len(work)
    return (max(work) - min(work)) / avg if avg else 0.0


# ----------------------------------------------------------------------------
# Task 3.2 + 3.4: sweep + imbalance
# ----------------------------------------------------------------------------
def sweep():
    img = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    render_rows(img, 0, 4, WIDTH, HEIGHT, MAX_ITER)  # JIT warm-up
    print(f"Mandelbrot {WIDTH}x{HEIGHT}, max_iter={MAX_ITER}, runs/cell={RUNS}")

    # классический static (блоки N/P) - отдельно, без chunk
    results = {}       # (policy, P, C) -> (mean_time, imbalance)
    raw_rows = []
    per_thread_rows = []

    def bench(policy, P, C, label_C):
        ts, imbs, last_work, last_rows = [], [], None, None
        for r in range(RUNS):
            t, w, rr = run_schedule(policy, P, C, img)
            ts.append(t)
            imbs.append(imbalance(w))
            last_work, last_rows = w, rr
            raw_rows.append((policy, P, label_C, r + 1, f"{t:.6f}", f"{imbalance(w):.4f}"))
        for tid in range(P):
            per_thread_rows.append((policy, P, label_C, tid, last_work[tid], last_rows[tid]))
        return statistics.mean(ts), statistics.mean(imbs)

    print(f"\n{'policy':>8} {'P':>3} {'C':>6} | {'mean time, s':>12} | {'imbalance':>9}")
    for P in THREADS:
        m, im = bench("static", P, None, "block")
        results[("static-block", P, "block")] = (m, im)
        print(f"{'static':>8} {P:>3} {'block':>6} | {m:12.4f} | {im:9.3f}")
        for C in CHUNKS:
            for pol in POLICIES:
                m, im = bench(pol, P, C, C)
                results[(pol, P, C)] = (m, im)
                print(f"{pol:>8} {P:>3} {C:>6} | {m:12.4f} | {im:9.3f}")

    # CSV
    path = os.path.join(RESULTS_DIR, "task3_2_sweep_raw.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "threads", "chunk", "run", "time_s", "imbalance"])
        w.writerows(raw_rows)
    path2 = os.path.join(RESULTS_DIR, "task3_2_sweep_mean.csv")
    with open(path2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "threads", "chunk", "mean_time_s", "mean_imbalance"])
        for (pol, P, C), (m, im) in results.items():
            w.writerow([pol, P, C, f"{m:.6f}", f"{im:.4f}"])
    path3 = os.path.join(RESULTS_DIR, "task3_4_per_thread_work.csv")
    with open(path3, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "threads", "chunk", "thread", "iterations_work", "rows"])
        w.writerows(per_thread_rows)
    print(f"\nCSV saved -> {RESULTS_DIR}")
    return results


# ----------------------------------------------------------------------------
# Task 3.3: heatmaps
# ----------------------------------------------------------------------------
def plot(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен -> графики пропущены")
        return

    def heat(metric_idx, title, fname, fmt):
        fig, axes = plt.subplots(1, len(POLICIES), figsize=(5 * len(POLICIES), 4))
        for ax, pol in zip(axes, POLICIES):
            data = np.array([[results[(pol, P, C)][metric_idx] for C in CHUNKS] for P in THREADS])
            im = ax.imshow(data, cmap="viridis_r" if metric_idx == 0 else "magma_r", aspect="auto")
            ax.set_xticks(range(len(CHUNKS)), CHUNKS)
            ax.set_yticks(range(len(THREADS)), THREADS)
            ax.set_xlabel("Chunk size C")
            ax.set_ylabel("Threads P")
            ax.set_title(pol)
            for i in range(len(THREADS)):
                for j in range(len(CHUNKS)):
                    ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                            color="white", fontsize=9)
            fig.colorbar(im, ax=ax)
        fig.suptitle(title)
        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, fname)
        fig.savefig(out, dpi=200)
        print(f"heatmap saved -> {out}")

    heat(0, "Execution time (s) vs chunk size and threads", "task3_3_heatmap_time.png", ".2f")
    heat(1, "Load imbalance (max-min)/avg", "task3_4_heatmap_imbalance.png", ".2f")

    # grouped bar chart для P=4 (альтернатива heatmap из задания)
    P = 4 if 4 in THREADS else THREADS[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    x = np.arange(len(CHUNKS))
    for k, pol in enumerate(POLICIES):
        ax.bar(x + k * width, [results[(pol, P, C)][0] for C in CHUNKS], width, label=pol)
    ax.set_xticks(x + width, CHUNKS)
    ax.set_xlabel("Chunk size C")
    ax.set_ylabel("Time, s")
    ax.set_title(f"Scheduling policies, P={P}")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "task3_3_bars_P4.png")
    fig.savefig(out, dpi=200)
    print(f"bar chart saved -> {out}")

    # сама картинка фрактала (для отчёта)
    img = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    render_rows(img, 0, HEIGHT, WIDTH, HEIGHT, MAX_ITER)
    plt.figure(figsize=(6, 3.4))
    plt.imshow(img, cmap="inferno")
    plt.axis("off")
    out = os.path.join(RESULTS_DIR, "mandelbrot.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"fractal saved -> {out}")

    # рабочая нагрузка по строкам - наглядно показывает spatial imbalance (Q3.2)
    row_cost = img.sum(axis=1)
    plt.figure(figsize=(7, 3.5))
    plt.plot(row_cost)
    plt.xlabel("Row index")
    plt.ylabel("Total escape iterations")
    plt.title("Cost per row (non-uniform workload)")
    plt.grid(alpha=0.3)
    out = os.path.join(RESULTS_DIR, "task3_row_cost.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"row cost saved -> {out}")


if __name__ == "__main__":
    print(f"Logical CPUs: {os.cpu_count()}")
    res = sweep()
    plot(res)
