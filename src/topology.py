# Topological snow filter based on 0-dimensional persistent homology (H0).
#
# Idea: grow a "scale" from 0 upwards and watch points merge into connected
# components. Real surfaces are sampled densely, so their points merge into big
# components almost immediately. Snowflake returns float alone in the air and
# only join something at a much larger scale. The scale where things merge is
# exactly what the H0 persistence diagram records.

import numpy as np
from scipy.spatial import cKDTree

from simulate import SENSOR, AZ_STEP

K_NEIGHBOURS = 10     # each point is connected to its 10 nearest neighbours
MIN_CLUSTER = 3       # a "real" structure needs at least this many points


def knn_graph(points, k=K_NEIGHBOURS):
    # edges between every point and its k nearest neighbours, each edge listed once
    tree = cKDTree(points)
    dist, idx = tree.query(points, k=k + 1)   # first neighbour is the point itself
    i = np.repeat(np.arange(len(points)), k)
    j = idx[:, 1:].ravel()
    d = dist[:, 1:].ravel()

    # (i, j) and (j, i) are the same edge: sort each pair, then drop duplicates
    pairs = np.sort(np.stack([i, j], axis=1), axis=1)
    pairs, first = np.unique(pairs, axis=0, return_index=True)
    return pairs, d[first]


def edge_weights(points, pairs, dists):
    # distance measured in "beam spacings" instead of metres.
    # far away the lidar samples sparsely, so 30 cm is normal at 40 m but huge at 2 m.
    ranges = np.linalg.norm(points - SENSOR, axis=1)
    mean_range = (ranges[pairs[:, 0]] + ranges[pairs[:, 1]]) / 2
    spacing = mean_range * np.deg2rad(AZ_STEP)    # gap between two shots at that range
    return dists / spacing


def h0_persistence(n_points, pairs, weights, min_cluster=MIN_CLUSTER):
    # Kruskal's algorithm with union-find.
    # Every point is born at scale 0. Going through edges from short to long, an
    # edge that joins two different components kills one of them: its death time
    # is the edge weight. These death times are the H0 persistence diagram.
    #
    # On the way we also record, for every point, the scale at which it first
    # belongs to a component with at least min_cluster points ("join scale").
    parent = np.arange(n_points)
    size = np.ones(n_points, dtype=int)
    waiting = {i: [i] for i in range(n_points)}   # points still in small components
    join_scale = np.full(n_points, np.inf)
    deaths = []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]         # path halving keeps trees flat
            a = parent[a]
        return a

    for e in np.argsort(weights):
        a, b = find(pairs[e, 0]), find(pairs[e, 1])
        if a == b:
            continue                              # already connected, no change in H0
        w = weights[e]
        deaths.append(w)

        if size[a] < size[b]:
            a, b = b, a                           # attach smaller tree to bigger one
        parent[b] = a
        size[a] += size[b]

        pending = waiting.pop(a, []) + waiting.pop(b, [])
        if size[a] >= min_cluster:
            join_scale[pending] = w               # these points just joined a real structure
        else:
            waiting[a] = pending

    return np.array(deaths), join_scale


def otsu_threshold(values, bins=200):
    # split values into two groups so that each group is as tight as possible.
    # we use it on log(death times): surface merges vs snow merges.
    hist, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    p = hist / hist.sum()

    w0 = np.cumsum(p)                 # weight of the lower group for every cut
    w1 = 1 - w0
    mu0 = np.cumsum(p * centres) / np.maximum(w0, 1e-12)
    mu_total = np.sum(p * centres)
    mu1 = (mu_total - np.cumsum(p * centres)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (mu0 - mu1) ** 2  # separation between the two groups
    return centres[np.argmax(between)]


def topo_filter(points, k=K_NEIGHBOURS, min_cluster=MIN_CLUSTER, threshold=None):
    # returns a boolean mask: True = point looks like noise.
    # if threshold is None it is picked automatically from the persistence diagram.
    pairs, dists = knn_graph(points, k)
    weights = edge_weights(points, pairs, dists)
    deaths, join_scale = h0_persistence(len(points), pairs, weights, min_cluster)

    if threshold is None:
        threshold = 10 ** otsu_threshold(np.log10(deaths))
    is_noise = join_scale > threshold
    return is_noise, deaths, threshold


if __name__ == "__main__":
    from simulate import simulate_scan, SNOW

    points, labels = simulate_scan(snow_rate=0.1, seed=0)
    is_noise, deaths, threshold = topo_filter(points)
    is_snow = labels == SNOW

    print("threshold:", round(threshold, 2))
    print("snow points:", is_snow.sum(), "caught:", (is_noise & is_snow).sum())
    print("real points removed by mistake:", (is_noise & ~is_snow).sum(), "of", (~is_snow).sum())