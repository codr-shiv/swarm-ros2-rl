"""360° lidar (TurtleBot3 LDS-01: 3.5 m range), vectorised over all rays."""
import numpy as np

LIDAR_RANGE = 3.5      # m
NUM_RAYS = 360


class Lidar:

    def __init__(self, resolution):
        steps = int(LIDAR_RANGE / resolution)
        angles = np.linspace(0, 2 * np.pi, NUM_RAYS, endpoint=False)
        t = np.arange(1, steps + 1) * 0.999          # sub-cell steps so no cell is skipped
        self.drow = np.rint(np.sin(angles)[:, None] * t).astype(np.int32)   # (rays, steps)
        self.dcol = np.rint(np.cos(angles)[:, None] * t).astype(np.int32)
        self.steps = steps

    def scan(self, belief, truth, row, col):
        """Copy the ground truth into *belief* for every cell a ray reaches (stops at walls)."""
        h, w = truth.shape
        rows = np.clip(row + self.drow, 0, h - 1)
        cols = np.clip(col + self.dcol, 0, w - 1)
        hit = truth[rows, cols] != 0
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), self.steps - 1)
        seen = np.arange(self.steps)[None, :] <= first[:, None]
        belief[rows[seen], cols[seen]] = truth[rows[seen], cols[seen]]
        belief[row, col] = truth[row, col]
