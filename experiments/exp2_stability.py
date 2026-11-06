# Experiment 2: does the stability theorem hold in practice?
#
# Stability theorem (Cohen-Steiner, Edelsbrunner, Harer 2007; Chazal et al. 2009):
# if every point of a cloud moves by at most eps, the persistence diagram moves
# by at most eps in bottleneck distance (with balls of radius r, i.e. an edge
# appears at r = length / 2). We move points by exactly eps in random directions
# and measure how far the H0 diagram actually moves.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, maximum_bipartite_matching
from scipy.spatial.distance import pdist, squareform

from simulate import simulate_scan

N_POINTS = 600                      # bottleneck matching is slow, so we use a subset
EPSILONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
REPEATS = 5


def h0_diagram(points):
    # for H0 every point is born at 0 and the deaths are the edges of the
    # minimum spanning tree (that's what Kruskal finds). radius = length / 2.
    dist = squareform(pdist(points))
    mst = minimum_spanning_tree(csr_matrix(dist))
    deaths = np.sort(mst.data) / 2
    return np.stack([np.zeros_like(deaths), deaths], axis=1)


def bottleneck(d1, d2):
    # bottleneck distance between two diagrams: the smallest delta such that every
    # point of one diagram can be matched to a point of the other (or to the
    # diagonal) while moving at most delta in the max-norm.
    # we binary search delta and check for a perfect matching each time.
    n, m = len(d1), len(d2)
    cost = np.max(np.abs(d1[:, None, :] - d2[None, :, :]), axis=2)   # point to point
    diag1 = (d1[:, 1] - d1[:, 0]) / 2                                # point to diagonal
    diag2 = (d2[:, 1] - d2[:, 0]) / 2
    candidates = np.unique(np.concatenate([cost.ravel(), diag1, diag2, [0.0]]))

    def has_perfect_matching(delta):
        # left side: points of d1, then m diagonal copies; right side: points of d2, then n copies
        rows, cols = np.nonzero(cost <= delta)
        r1 = np.flatnonzero(diag1 <= delta)                   # d1 point -> its diagonal copy
        r2 = np.flatnonzero(diag2 <= delta)                   # diagonal copy -> d2 point
        dd_r, dd_c = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")  # diagonal <-> diagonal
        all_r = np.concatenate([rows, r1, n + r2, n + dd_r.ravel()])
        all_c = np.concatenate([cols, m + r1, r2, m + dd_c.ravel()])
        graph = csr_matrix((np.ones(len(all_r)), (all_r, all_c)), shape=(n + m, m + n))
        match = maximum_bipartite_matching(graph, perm_type="column")
        return np.all(match >= 0)

    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if has_perfect_matching(candidates[mid]):
            hi = mid
        else:
            lo = mid + 1
    return candidates[lo]


def main():
    rng = np.random.default_rng(0)
    points, _ = simulate_scan(snow_rate=0.0, seed=0)
    base = points[rng.choice(len(points), N_POINTS, replace=False)]
    d_base = h0_diagram(base)

    rows = []
    for eps in EPSILONS:
        for rep in range(REPEATS):
            # move every point by exactly eps in a random direction
            direction = rng.normal(size=base.shape)
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            moved = base + eps * direction
            d = bottleneck(d_base, h0_diagram(moved))
            rows.append({"eps": eps, "repeat": rep, "bottleneck": d, "bound_holds": d <= eps + 1e-12})
        b = [r["bottleneck"] for r in rows if r["eps"] == eps]
        print(f"eps = {eps:6.3f}   bottleneck: mean {np.mean(b):.4f}, max {np.max(b):.4f}   (bound: {eps})")

    print("bound held in", sum(r["bound_holds"] for r in rows), "of", len(rows), "runs")

    # optional cross-check with GUDHI if it is installed
    try:
        import gudhi
        moved = base + 0.05 * rng.normal(size=base.shape) / np.sqrt(3)
        print("our bottleneck:", round(bottleneck(d_base, h0_diagram(moved)), 5),
              "| gudhi:", round(gudhi.bottleneck_distance(d_base, h0_diagram(moved)), 5))
    except ImportError:
        print("gudhi not installed, skipping the cross-check")

    with open(ROOT / "results" / "exp2_stability.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6, 5))
    eps = np.array([r["eps"] for r in rows])
    b = np.array([r["bottleneck"] for r in rows])
    ax.scatter(eps, b, s=18, label="measured (5 random moves per eps)")
    line = np.array([min(EPSILONS), max(EPSILONS)])
    ax.plot(line, line, "k--", label="theoretical bound: d_B = eps")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eps: how far every point was moved (m)")
    ax.set_ylabel("bottleneck distance between H0 diagrams (m)")
    ax.set_title("Stability theorem, checked numerically")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "exp2_stability.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()