# Experiment 3: how much can we downsample before the topological filter breaks?
#
# Part A: why downsampling matters at all. The full Vietoris-Rips complex grows
# much faster than the number of points, so building it on a whole scan is not
# practical. We count its simplices (vertices, edges, triangles) for growing subsets.
#
# Part B: we keep only a fraction of the points (random, voxel grid or farthest
# point sampling), run the filter on what is left, then give every original point
# the label of its nearest kept point. We measure quality and run time.

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from simulate import simulate_scan, SNOW
from topology import topo_filter
from metrics import scores

RIPS_RADIUS = 0.5                     # edge length used for the Rips complex in part A
SUBSET_SIZES = [500, 1000, 2000, 4000, 8000, 16000]
KEEP_FRACTIONS = [1.0, 0.5, 0.25, 0.1]
VOXEL_SIZES = [0.05, 0.1, 0.2, 0.4]
SEEDS = range(3)


def count_rips_simplices(points, radius):
    # vertices + edges + triangles of the Rips complex with edges of length <= radius.
    # triangles = number of 3-cycles in the graph = trace(A^3) / 6
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    n = len(points)
    a = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    a = a + a.T
    triangles = int((a @ a).multiply(a).sum() / 6)
    return n, len(pairs), triangles


def random_sample(points, fraction, rng):
    return rng.choice(len(points), int(len(points) * fraction), replace=False)


def voxel_sample(points, voxel):
    # keep the first point that falls into every voxel
    cells = np.floor(points / voxel).astype(np.int64)
    _, keep = np.unique(cells, axis=0, return_index=True)
    return np.sort(keep)


def farthest_point_sample(points, fraction, rng):
    # greedy: always add the point that is furthest from everything chosen so far.
    # spreads points evenly, but costs one pass over the cloud per chosen point.
    m = int(len(points) * fraction)
    chosen = np.empty(m, dtype=int)
    chosen[0] = rng.integers(len(points))
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    for i in range(1, m):
        chosen[i] = np.argmax(dist)
        dist = np.minimum(dist, np.linalg.norm(points - points[chosen[i]], axis=1))
    return chosen


def filter_with_downsampling(points, keep):
    # run the filter on the kept points, copy labels to the rest by nearest kept point
    noise_kept, _, _ = topo_filter(points[keep])
    _, nearest = cKDTree(points[keep]).query(points)
    return noise_kept[nearest]


def part_a():
    points, _ = simulate_scan(0.0, seed=0)
    near = points[np.linalg.norm(points[:, :2], axis=1) < 20]   # fixed area, so density grows with n
    rng = np.random.default_rng(0)
    rows = []
    print(f"Rips complex, edges <= {RIPS_RADIUS} m, points within 20 m of the sensor:")
    for n in SUBSET_SIZES + [len(near)]:
        if n > len(near):
            continue
        subset = near[rng.choice(len(near), n, replace=False)]
        start = time.perf_counter()
        v, e, t = count_rips_simplices(subset, RIPS_RADIUS)
        rows.append({"points": v, "edges": e, "triangles": t, "total": v + e + t,
                     "seconds": time.perf_counter() - start})
        print(f"  {v:6d} points -> {e:9d} edges, {t:11d} triangles")
    return rows


def part_b():
    rows = []
    for seed in SEEDS:
        pts, lab = simulate_scan(0.10, seed)
        is_snow = lab == SNOW
        rng = np.random.default_rng(seed)

        # (family, readable name, function that returns the indices of kept points)
        settings = [("full", "full scan", lambda: np.arange(len(pts)))]
        for f in KEEP_FRACTIONS[1:]:
            settings.append(("random", f"random {f:.0%}", lambda f=f: random_sample(pts, f, rng)))
            settings.append(("farthest point", f"farthest point {f:.0%}",
                             lambda f=f: farthest_point_sample(pts, f, rng)))
        for v in VOXEL_SIZES:
            settings.append(("voxel", f"voxel {v} m", lambda v=v: voxel_sample(pts, v)))

        for family, name, pick in settings:
            start = time.perf_counter()
            keep = pick()
            t_sample = time.perf_counter() - start
            start = time.perf_counter()
            noise = filter_with_downsampling(pts, keep)
            t_filter = time.perf_counter() - start
            s = scores(is_snow, noise)
            rows.append({"seed": seed, "method": family, "setting": name,
                         "kept_fraction": len(keep) / len(pts),
                         "f1": s["f1"], "kept_real": s["kept_real"],
                         "sample_seconds": t_sample, "filter_seconds": t_filter})
        print(f"done: seed {seed}")

    print(f"\n{'setting':22s} {'kept pts':>9s} {'F1':>7s} {'kept real':>10s} {'filter s':>9s} {'sample s':>9s}")
    for setting in dict.fromkeys(r["setting"] for r in rows):
        r = [x for x in rows if x["setting"] == setting]
        print(f"{setting:22s} {np.mean([x['kept_fraction'] for x in r]):9.2f} "
              f"{np.mean([x['f1'] for x in r]):7.3f} {np.mean([x['kept_real'] for x in r]):10.4f} "
              f"{np.mean([x['filter_seconds'] for x in r]):9.2f} {np.mean([x['sample_seconds'] for x in r]):9.2f}")
    return rows


def plot(rows_a, rows_b):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    n = np.array([r["points"] for r in rows_a])
    axes[0].loglog(n, [r["edges"] for r in rows_a], "o-", label="edges")
    axes[0].loglog(n, [r["triangles"] for r in rows_a], "s-", label="triangles")
    axes[0].loglog(n, n, "k:", label="points (linear growth)")
    axes[0].set_xlabel("number of points (same 20 m area)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Rips complex size, edges <= {RIPS_RADIUS} m")
    axes[0].legend()

    for method, marker in [("random", "o"), ("farthest point", "s"), ("voxel", "^")]:
        r = [x for x in rows_b if x["method"] in (method, "full")]
        settings = list(dict.fromkeys(x["setting"] for x in r))
        fr = np.array([np.mean([x["kept_fraction"] for x in r if x["setting"] == st]) for st in settings])
        f1 = np.array([np.mean([x["f1"] for x in r if x["setting"] == st]) for st in settings])
        sec = np.array([np.mean([x["filter_seconds"] for x in r if x["setting"] == st]) for st in settings])
        order = np.argsort(fr)
        axes[1].plot(fr[order], f1[order], marker + "-", label=method)
        axes[2].plot(fr[order], sec[order], marker + "-", label=method)

    axes[1].set_xlabel("fraction of points kept")
    axes[1].set_ylabel("F1 on the full scan")
    axes[1].set_title("Snow removal after downsampling (heavy snow)")
    axes[1].legend()
    axes[2].set_xlabel("fraction of points kept")
    axes[2].set_ylabel("filter time, s (without sampling)")
    axes[2].set_title("Filter run time")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "exp3_downsampling.png", dpi=170)
    plt.close(fig)


def main():
    rows_a = part_a()
    rows_b = part_b()
    for name, rows in [("exp3_rips_growth.csv", rows_a), ("exp3_downsampling.csv", rows_b)]:
        with open(ROOT / "results" / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    plot(rows_a, rows_b)


if __name__ == "__main__":
    main()