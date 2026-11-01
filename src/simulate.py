import numpy as np
from contourpy import as_z_interp
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

""" different angles for lasers in LIDAR"""
elevations =  np.linspace(-16, 10, 32)
elevations_rad = np.deg2rad(elevations)


"""places in angles where our laser will take point -> 900 positions """
azimuths = np.arange(0, 360, 0.4)
azimuths_rad = np.deg2rad(azimuths)
"""32 rows 900 columns"""
L, R = np.meshgrid(elevations_rad, azimuths_rad, indexing='ij')
"L in rows and dublicate and vice versa for R"

dx = np.cos(L) * np.cos(R)
dy = np.cos(L) * np.sin(R)
dz = np.sin(L)

directions = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)

"""initial sensor location on the car"""
sensor = np.array([0.0, 0.0, 1.8])
dz = directions[:, 2]
t_ground = np.full(len(directions), np.inf)

down = dz < 0

t_ground[down] = -sensor[2]/ dz[down]
# Building walls along the street: left at y = +10, right at y = -10, 8 m tall
ray_dy = directions[:, 1]

# Left wall
t_left = np.full(len(directions), np.inf)
to_left = ray_dy > 0
t_left[to_left] = 10 / ray_dy[to_left]
z_left = sensor[2] + t_left * dz
t_left[(z_left < 0) | (z_left > 8)] = np.inf  # below ground or above the roof

# Right wall
t_right = np.full(len(directions), np.inf)
to_right = ray_dy < 0
t_right[to_right] = -10 / ray_dy[to_right]
z_right = sensor[2] + t_right * dz
t_right[(z_right < 0) | (z_right > 8)] = np.inf

# Each ray stops at the first surface it hits
t = np.minimum(t_ground, np.minimum(t_left, t_right))
t[t > 80] = np.inf  # beyond sensor range

surface = np.where(t == t_ground, 0, 1)  # 0 = ground, 1 = wall

hit = np.isfinite(t)
points = sensor + t[hit, None] * directions[hit]
surface = surface[hit]



if __name__ == "__main__":
    print("points:", points.shape)
    print("ground:", np.sum(surface == 0), "walls:", np.sum(surface == 1))
    print("max |y|:", np.abs(points[:, 1]).max())

    plt.figure(figsize=(12, 4))
    plt.scatter(points[surface == 0, 0], points[surface == 0, 1], s=0.3, label="ground")
    plt.scatter(points[surface == 1, 0], points[surface == 1, 1], s=0.3, label="walls")
    plt.scatter(0, 0, c="red", s=25, label="sensor")
    plt.xlim(-60, 60)
    plt.ylim(-15, 15)
    plt.xlabel("x, m (forward)")
    plt.ylabel("y, m (left)")
    plt.title("LiDAR scan: ground and walls, top view")
    legend = plt.legend()
    for handle in legend.legend_handles:
        handle.set_sizes([20])  # same marker size for every legend entry

    plt.savefig(ROOT / "figures" / "ground_walls.png", dpi=200)
    plt.show()



