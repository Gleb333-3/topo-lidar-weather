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
t_ground[t_ground > 80] = np.inf

hit = np.isfinite(t_ground)
points = sensor + t_ground[hit, None] * directions[hit]


if __name__ == "__main__":
    print(directions.shape)          # должно быть (28800, 3)
    print(directions[0])             # первый луч: примерно [0.96, 0, -0.28]
    print(np.allclose(np.linalg.norm(directions, axis=1), 1))
    print(dz.shape)
    print(dz.min(), dz.max())
    print(t_ground[:5])
    print(np.sum(down))  # сколько лучей смотрит вниз
    print(np.sum(np.isfinite(t_ground)))  # сколько попало в асфальт ближе 80 м
    print(np.min(t_ground))  # ближайшее попадание, около 6.5
    print(points.shape)
    print(np.allclose(points[:, 2], 0))
    print(points[0])

    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=0.2)
    plt.scatter(0, 0, c="red", s=20)
    plt.axis("equal")
    plt.xlabel("x, m (forward)")
    plt.ylabel("y, m (left)")
    plt.title("LiDAR ground hits, top view")
    plt.savefig(ROOT / "figures" / "ground_rings.png", dpi=200)

    plt.show()


