# Experiment 1: can H0 persistence tell snow apart from real objects?
#
# For light, moderate and heavy snow we simulate several scans and run four filters:
# SOR, ROR, DROR and our topological filter. The classic filters are tried with
# their usual default settings and also after tuning them on separate scans using
# the true labels (their best possible case). The topological filter is never
# tuned: its threshold comes from the persistence diagram of each scan.

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import csv
import numpy as np
import matplotlib.pyplot as plt

from simulate import simulate_scan, SNOW, plot_top_view
from filters import sor, ror, dror
from topology import topo_filter
from metrics import scores

SNOW_RATES = {"light": 0.02, "moderate": 0.05, "heavy": 0.10}
TEST_SEEDS = range(10)          # scans used for the results
TUNE_SEEDS = range(100, 103)    # different scans, only used to tune the baselines

# parameter grids for tuning the baselines
GRIDS = {
    "SOR": (sor, [dict(k=k, std_ratio=s) for k in [5, 10, 20] for s in [0.5, 1, 2, 3]]),
    "ROR": (ror, [dict(radius=r, min_neighbours=m) for r in [0.1, 0.2, 0.3, 0.5, 1.0] for m in [1, 2, 3, 5]]),
    "DROR": (dror, [dict(beta=b, min_neighbours=m) for b in [1, 2, 3, 5, 8] for m in [1, 2, 3, 5]]),
}


def tune(fn, grid):
    # pick the parameters with the best mean F1 over all snow levels on the tuning scans
    scans = [simulate_scan(rate, seed) for rate in SNOW_RATES.values() for seed in TUNE_SEEDS]
    best_f1, best_params = -1, None
    for params in grid:
        f1 = np.mean([scores(lab == SNOW, fn(pts, **params))["f1"] for pts, lab in scans])
        if f1 > best_f1:
            best_f1, best_params = f1, params
    return best_params


def main():
    print("tuning baselines on separate scans...")
    tuned = {name: tune(fn, grid) for name, (fn, grid) in GRIDS.items()}
    for name, params in tuned.items():
        print(f"  {name}: {params}")

    methods = {
        "SOR (default)": lambda p: sor(p),
        "ROR (default)": lambda p: ror(p),
        "DROR (default)": lambda p: dror(p),
        "SOR (tuned)": lambda p: sor(p, **tuned["SOR"]),
        "ROR (tuned)": lambda p: ror(p, **tuned["ROR"]),
        "DROR (tuned)": lambda p: dror(p, **tuned["DROR"]),
        "Topological (ours, no tuning)": lambda p: topo_filter(p)[0],
    }

    rows = []
    for level, rate in SNOW_RATES.items():
        for seed in TEST_SEEDS:
            pts, lab = simulate_scan(rate, seed)
            is_snow = lab == SNOW
            far_ped = np.hypot(pts[:, 0] - 55, pts[:, 1] - 6) < 0.5
            for name, fn in methods.items():
                start = time.perf_counter()
                noise = fn(pts)
                seconds = time.perf_counter() - start
                s = scores(is_snow, noise)
                rows.append({"snow": level, "seed": seed, "method": name, **s,
                             "far_ped_kept": int(np.sum(far_ped & ~noise)),
                             "far_ped_total": int(np.sum(far_ped)),
                             "seconds": seconds})
        print(f"done: {level} snow")

    # save every run
    out = ROOT / "results" / "exp1_all_runs.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # summary table: mean +- std over seeds
    summary = []
    print(f"\n{'method':32s} {'snow':9s} {'precision':>10s} {'recall':>8s} {'F1':>13s} {'kept real':>10s}")
    for name in methods:
        for level in SNOW_RATES:
            r = [x for x in rows if x["method"] == name and x["snow"] == level]
            f1 = np.array([x["f1"] for x in r])
            line = {"method": name, "snow": level,
                    "precision": np.mean([x["precision"] for x in r]),
                    "recall": np.mean([x["recall"] for x in r]),
                    "f1_mean": f1.mean(), "f1_std": f1.std(),
                    "kept_real": np.mean([x["kept_real"] for x in r]),
                    "far_ped_kept": sum(x["far_ped_kept"] for x in r),
                    "far_ped_total": sum(x["far_ped_total"] for x in r),
                    "seconds": np.mean([x["seconds"] for x in r])}
            summary.append(line)
            print(f"{name:32s} {level:9s} {line['precision']:10.3f} {line['recall']:8.3f} "
                  f"{line['f1_mean']:6.3f} ± {line['f1_std']:.3f} {line['kept_real']:10.4f}")

    with open(ROOT / "results" / "exp1_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    make_main_figure()
    make_f1_figure(summary, list(methods))
    robustness(tuned["DROR"])


def robustness(dror_params):
    # what happens when conditions change after DROR was tuned?
    # we change one thing in the simulator at a time and keep every filter setting fixed.
    import simulate
    changes = {
        "baseline": {},
        "snow further from sensor (15 m)": {"SNOW_DIST": 15.0},
        "snow closer to sensor (2.5 m)": {"SNOW_DIST": 2.5},
        "noisier sensor (5 cm)": {"RANGE_NOISE": 0.05},
    }
    rows = []
    print(f"\n{'condition':34s} {'snow':9s} {'DROR (tuned)':>13s} {'Topological':>12s}")
    for cond, new_values in changes.items():
        old_values = {k: getattr(simulate, k) for k in new_values}
        for k, v in new_values.items():
            setattr(simulate, k, v)
        for level, rate in SNOW_RATES.items():
            f1_d, f1_t = [], []
            for seed in range(5):
                pts, lab = simulate.simulate_scan(rate, seed)
                is_snow = lab == SNOW
                f1_d.append(scores(is_snow, dror(pts, **dror_params))["f1"])
                f1_t.append(scores(is_snow, topo_filter(pts)[0])["f1"])
            rows.append({"condition": cond, "snow": level,
                         "dror_tuned_f1": np.mean(f1_d), "topological_f1": np.mean(f1_t)})
            print(f"{cond:34s} {level:9s} {np.mean(f1_d):13.3f} {np.mean(f1_t):12.3f}")
        for k, v in old_values.items():
            setattr(simulate, k, v)

    with open(ROOT / "results" / "exp1_robustness.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def make_main_figure():
    # the picture for the top of the README:
    # heavy-snow scan, the same scan after our filter, and the persistence diagrams
    pts_clear, _ = simulate_scan(0.0, seed=0)
    pts, lab = simulate_scan(0.10, seed=0)
    noise, deaths, threshold = topo_filter(pts)
    _, deaths_clear, _ = topo_filter(pts_clear)

    fig = plt.figure(figsize=(13, 9))
    ax1 = fig.add_subplot(3, 1, 1)
    plot_top_view(ax1, pts, lab, "Heavy snow: ground truth (brown = snow)")
    ax2 = fig.add_subplot(3, 1, 2)
    kept = ~noise
    ax2.scatter(pts[kept, 0], pts[kept, 1], s=0.5, c="tab:blue", label="kept")
    ax2.scatter(pts[noise, 0], pts[noise, 1], s=0.5, c="tab:red", label="removed as noise")
    ax2.scatter(0, 0, c="black", marker="^", s=40)
    ax2.set_aspect("equal")
    ax2.set_xlim(-30, 60)
    ax2.set_ylim(-12, 12)
    ax2.set_xlabel("x, m (forward)")
    ax2.set_ylabel("y, m (left)")
    ax2.set_title("After the topological filter (red = removed)")
    leg = ax2.legend(loc="upper right")
    for h in leg.legend_handles:
        h.set_sizes([20])

    ax3 = fig.add_subplot(3, 1, 3)
    bins = np.logspace(np.log10(min(deaths.min(), deaths_clear.min()) + 1e-3),
                       np.log10(max(deaths.max(), deaths_clear.max())), 80)
    ax3.hist(deaths_clear, bins=bins, alpha=0.6, label="clear weather")
    ax3.hist(deaths, bins=bins, alpha=0.6, label="heavy snow")
    ax3.axvline(threshold, color="black", ls="--", label=f"automatic threshold = {threshold:.1f}")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("death time of H0 features (scale, in beam spacings)")
    ax3.set_ylabel("count")
    ax3.set_title("H0 persistence diagram as a histogram of death times: snow adds a second, long-lived group")
    ax3.legend()

    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "exp1_main.png", dpi=170)
    plt.close(fig)


def make_f1_figure(summary, names):
    levels = list(SNOW_RATES)
    x = np.arange(len(levels))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, name in enumerate(names):
        means = [next(s["f1_mean"] for s in summary if s["method"] == name and s["snow"] == lv) for lv in levels]
        stds = [next(s["f1_std"] for s in summary if s["method"] == name and s["snow"] == lv) for lv in levels]
        ax.bar(x + i * width - 0.4 + width / 2, means, width, yerr=stds, capsize=2, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lv} snow" for lv in levels])
    ax.set_ylabel("F1 (snow detection)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Snow removal quality, mean ± std over 10 scans")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "exp1_f1.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()