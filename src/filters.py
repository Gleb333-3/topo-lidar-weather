# Classic (non-topological) outlier filters used as baselines.
# Each one returns a boolean mask: True = point looks like noise.

import numpy as np
from scipy.spatial import cKDTree

from simulate import SENSOR, AZ_STEP


def sor(points, k=10, std_ratio=1.0):
    # Statistical Outlier Removal: a point is noise if its mean distance to the
    # k nearest neighbours is much larger than average over the whole cloud.
    dist, _ = cKDTree(points).query(points, k=k + 1)
    mean_dist = dist[:, 1:].mean(axis=1)
    return mean_dist > mean_dist.mean() + std_ratio * mean_dist.std()


def ror(points, radius=0.5, min_neighbours=3):
    # Radius Outlier Removal: noise if fewer than min_neighbours points within a fixed radius.
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=radius, return_length=True) - 1
    return counts < min_neighbours


def dror(points, beta=3.0, min_radius=0.04, min_neighbours=3):
    # Dynamic Radius Outlier Removal (Charron et al., 2018), made for snow.
    # Same as ROR but the search radius grows with distance from the sensor,
    # because far away the points are naturally further apart.
    ranges = np.linalg.norm(points[:, :2] - SENSOR[:2], axis=1)
    radius = np.maximum(min_radius, beta * ranges * np.deg2rad(AZ_STEP))
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=radius, return_length=True) - 1
    return counts < min_neighbours