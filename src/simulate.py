# Simple LiDAR simulator: a street with walls, cars, people and poles, plus optional snow.
# Every point gets a label so we know which points are real and which are snow.
# Axes: x forward, y left, z up (metres). Origin is on the ground under the sensor.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# sensor settings (roughly a 32-beam car lidar)
N_BEAMS = 32
ELEV_MIN = -16.0      # degrees
ELEV_MAX = 10.0
AZ_STEP = 0.4         # degrees between shots
SENSOR = np.array([0.0, 0.0, 1.8])
MAX_RANGE = 80.0
RANGE_NOISE = 0.02    # 2 cm noise on every measurement

# street
WALL_DIST = 10.0      # walls at y = 10 and y = -10
WALL_HEIGHT = 8.0

# cars: (x, y, length, width, height)
CARS = [
    (14.0, 0.0, 4.5, 1.8, 1.5),
    (25.0, -4.0, 4.5, 1.8, 1.5),
    (-12.0, 3.5, 4.5, 1.8, 1.5),
    (45.0, 3.5, 4.5, 1.8, 1.5),
]

# people and poles: (x, y) of the centre
PEDESTRIANS = [(8.0, -7.5), (20.0, 7.5), (33.0, -7.0), (-6.0, -7.5), (55.0, 6.0)]
POLES = [(5.0, 8.5), (25.0, 8.5), (45.0, 8.5), (15.0, -8.5), (35.0, -8.5), (-10.0, -8.5)]
PED_RADIUS, PED_HEIGHT = 0.3, 1.75
POLE_RADIUS, POLE_HEIGHT = 0.15, 6.0

# snow: most detected flakes are close to the sensor
SNOW_DIST = 6.0       # average distance of a snow point
SNOW_MIN = 0.5        # sensor can't see closer than this

LABELS = ["ground", "walls", "cars", "pedestrians", "poles", "snow"]
SNOW = 5


def make_rays():
    # one direction (unit vector) for every laser at every azimuth
    elev = np.deg2rad(np.linspace(ELEV_MIN, ELEV_MAX, N_BEAMS))
    az = np.deg2rad(np.arange(0, 360, AZ_STEP))
    elev_grid, az_grid = np.meshgrid(elev, az, indexing="ij")

    dx = np.cos(elev_grid) * np.cos(az_grid)
    dy = np.cos(elev_grid) * np.sin(az_grid)
    dz = np.sin(elev_grid)
    return np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)


def hit_ground(origin, dirs):
    t = np.full(len(dirs), np.inf)
    down = dirs[:, 2] < 0
    t[down] = -origin[2] / dirs[down, 2]
    return t


def hit_wall(origin, dirs, y_wall, height):
    t = np.full(len(dirs), np.inf)
    if y_wall > origin[1]:
        towards = dirs[:, 1] > 0   # left wall, ray must go left
    else:
        towards = dirs[:, 1] < 0   # right wall, ray must go right
    t[towards] = (y_wall - origin[1]) / dirs[towards, 1]

    # too high = over the roof, below 0 = it hit the road first
    z = origin[2] + t * dirs[:, 2]
    t[(z < 0) | (z > height)] = np.inf
    return t


def hit_box(origin, dirs, box_min, box_max):
    # slab method: find where the ray enters/leaves the box along x, y and z.
    # it's inside the box only when it's inside all three at once.
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (box_min - origin) / dirs
        t2 = (box_max - origin) / dirs
    t_in = np.max(np.minimum(t1, t2), axis=1)
    t_out = np.min(np.maximum(t1, t2), axis=1)
    hit = (t_in <= t_out) & (t_out > 0)
    return np.where(hit, np.maximum(t_in, 0), np.inf)


def hit_cylinder(origin, dirs, cx, cy, radius, height):
    # from above the cylinder is a circle -> quadratic equation for t.
    # only the side is modelled, not the top.
    fx = origin[0] - cx
    fy = origin[1] - cy
    a = dirs[:, 0] ** 2 + dirs[:, 1] ** 2
    b = 2 * (fx * dirs[:, 0] + fy * dirs[:, 1])
    c = fx ** 2 + fy ** 2 - radius ** 2
    disc = b ** 2 - 4 * a * c

    with np.errstate(invalid="ignore"):
        t = (-b - np.sqrt(disc)) / (2 * a)   # nearest root, nan if no hit
    z = origin[2] + t * dirs[:, 2]
    hit = (disc >= 0) & (t > 0) & (z >= 0) & (z <= height)
    return np.where(hit, t, np.inf)


def cast_rays(origin, dirs):
    # distance to the first thing each ray hits + what that thing is
    t_ground = hit_ground(origin, dirs)
    t_walls = np.minimum(hit_wall(origin, dirs, WALL_DIST, WALL_HEIGHT),
                         hit_wall(origin, dirs, -WALL_DIST, WALL_HEIGHT))

    t_cars = np.full(len(dirs), np.inf)
    for x, y, length, width, height in CARS:
        box_min = np.array([x - length / 2, y - width / 2, 0])
        box_max = np.array([x + length / 2, y + width / 2, height])
        t_cars = np.minimum(t_cars, hit_box(origin, dirs, box_min, box_max))

    t_peds = np.full(len(dirs), np.inf)
    for x, y in PEDESTRIANS:
        t_peds = np.minimum(t_peds, hit_cylinder(origin, dirs, x, y, PED_RADIUS, PED_HEIGHT))

    t_poles = np.full(len(dirs), np.inf)
    for x, y in POLES:
        t_poles = np.minimum(t_poles, hit_cylinder(origin, dirs, x, y, POLE_RADIUS, POLE_HEIGHT))

    # rows = object types, columns = rays. closest one wins
    all_t = np.stack([t_ground, t_walls, t_cars, t_peds, t_poles])
    labels = np.argmin(all_t, axis=0)
    t = np.min(all_t, axis=0)
    return t, labels


def add_snow(t, labels, snow_rate, rng):
    # snow_rate = fraction of rays that hit a snowflake (0.02 light, 0.05 medium, 0.1 heavy)
    n = len(t)
    hits_flake = rng.random(n) < snow_rate
    flake_dist = SNOW_MIN + rng.exponential(SNOW_DIST, size=n)

    # the flake only counts if it's in front of the real object
    snow = hits_flake & (flake_dist < t)
    t = np.where(snow, flake_dist, t)
    labels = np.where(snow, SNOW, labels)
    return t, labels


def simulate_scan(snow_rate=0.0, seed=0):
    # returns points (N, 3) and labels (N,). same seed = same scan
    rng = np.random.default_rng(seed)
    dirs = make_rays()

    t, labels = cast_rays(SENSOR, dirs)
    t, labels = add_snow(t, labels, snow_rate, rng)

    t = t + rng.normal(0, RANGE_NOISE, size=len(t))
    t[t > MAX_RANGE] = np.inf

    hit = np.isfinite(t)
    points = SENSOR + t[hit, None] * dirs[hit]
    return points, labels[hit]


def plot_top_view(ax, points, labels, title):
    for i, name in enumerate(LABELS):
        mask = labels == i
        if mask.any():
            ax.scatter(points[mask, 0], points[mask, 1], s=0.5, label=name)
    ax.scatter(0, 0, c="red", marker="^", s=40, label="sensor")
    ax.set_aspect("equal")
    ax.set_xlim(-30, 60)
    ax.set_ylim(-12, 12)
    ax.set_xlabel("x, m (forward)")
    ax.set_ylabel("y, m (left)")
    ax.set_title(title)


if __name__ == "__main__":
    clear_pts, clear_lab = simulate_scan(snow_rate=0.0, seed=0)
    snow_pts, snow_lab = simulate_scan(snow_rate=0.1, seed=0)

    for name, lab in [("clear", clear_lab), ("heavy snow", snow_lab)]:
        counts = {LABELS[i]: int(np.sum(lab == i)) for i in range(len(LABELS))}
        print(name, len(lab), counts)

    # how many points land on the far pedestrian at (55, 6)
    far = np.hypot(clear_pts[:, 0] - 55, clear_pts[:, 1] - 6) < 0.5
    print("far pedestrian points:", np.sum(far))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plot_top_view(axes[0], clear_pts, clear_lab, "Clear weather")
    plot_top_view(axes[1], snow_pts, snow_lab, "Heavy snow (10% of rays hit a flake)")
    handles, names = axes[1].get_legend_handles_labels()
    legend = fig.legend(handles, names, loc="center right")
    for h in legend.legend_handles:
        h.set_sizes([20])
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(ROOT / "figures" / "scan_clear_vs_snow.png", dpi=200)
    plt.show()