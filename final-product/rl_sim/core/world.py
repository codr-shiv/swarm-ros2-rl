"""
Ground-truth 2D arena, identical to the Gazebo arena with the same number.

generate_random_world() (used by spawn_two_turtlebots.launch.py) writes an
SDF file; this module parses that file and rasterises its walls and
obstacles onto a 0.05 m grid (the SLAM map resolution), so the 2D simulator
and Gazebo (GAZEBO_WORLD_SEED=<number>) contain exactly the same arena.
"""
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import cv2  # type: ignore[import-untyped]
import numpy as np

from multi_robot_exploration.generate_random_world import generate_random_world

DEFAULT_WORLD = 42         # benchmark arena used for training and evaluation
RESOLUTION = 0.05          # m per cell, same as slam_toolbox / map merge
HALF_EXTENT = 4.25         # m, covers the 8 m arena plus its walls
ROBOT_INFLATION = 0.20     # m, robot radius 0.105 + margin (Nav2 inflation)
FREE, OCCUPIED = 0, 100


@dataclass
class World:
    world_id: int
    grid: np.ndarray        # (h, w) uint8, FREE / OCCUPIED ground truth
    nav_free: np.ndarray    # (h, w) bool, where the robot centre may be
    component: np.ndarray   # (h, w) int32, connected regions of nav_free
    spawns: list            # [(row, col), (row, col)]
    resolution: float = RESOLUTION
    origin: float = -HALF_EXTENT

    def to_cell(self, x, y):
        return (int((y - self.origin) / self.resolution), int((x - self.origin) / self.resolution))

    def to_world(self, row, col):
        return (self.origin + (col + 0.5) * self.resolution,
                self.origin + (row + 0.5) * self.resolution)


def _floats(text):
    return [float(v) for v in text.split()]


def _rasterise(sdf_path):
    n = int(round(2 * HALF_EXTENT / RESOLUTION))
    grid = np.zeros((n, n), dtype=np.uint8)

    def cell(v):
        return int(round((v + HALF_EXTENT) / RESOLUTION))

    root = ET.parse(sdf_path).getroot()
    for model in root.iter('model'):
        model_pose = model.find('pose')
        mx, my = (_floats(model_pose.text)[:2] if model_pose is not None else (0.0, 0.0))
        for link in model.iter('link'):
            link_pose = link.find('pose')
            lx, ly = (_floats(link_pose.text)[:2] if link_pose is not None else (0.0, 0.0))
            cx, cy = mx + lx, my + ly
            geom = link.find('collision/geometry')
            if geom is None:
                continue
            box = geom.find('box/size')
            cyl = geom.find('cylinder/radius')
            if box is not None:
                sx, sy = _floats(box.text)[:2]
                grid[cell(cy - sy / 2):cell(cy + sy / 2) + 1, cell(cx - sx / 2):cell(cx + sx / 2) + 1] = OCCUPIED
            elif cyl is not None:
                r = float(cyl.text)
                cv2.circle(grid, (cell(cx), cell(cy)), int(round(r / RESOLUTION)), OCCUPIED, -1)
    return grid


_CACHE = {}


def load_world(world_id=DEFAULT_WORLD):
    """Build (and cache) the arena that Gazebo builds for GAZEBO_WORLD_SEED=world_id."""
    if world_id in _CACHE:
        return _CACHE[world_id]
    path, spawns_xy = generate_random_world(seed=world_id)
    try:
        grid = _rasterise(path)
    finally:
        os.remove(path)

    r = int(np.ceil(ROBOT_INFLATION / RESOLUTION))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    nav_free = cv2.dilate((grid == OCCUPIED).astype(np.uint8), kernel) == 0
    _, component = cv2.connectedComponents(nav_free.astype(np.uint8), connectivity=8)

    world = World(world_id, grid, nav_free, component.astype(np.int32), [])
    for x, y in spawns_xy:
        row, col = world.to_cell(x, y)
        if not nav_free[row, col]:        # spawn too close to an obstacle: nearest free cell
            rr, cc = np.nonzero(nav_free)
            k = int(np.argmin((rr - row) ** 2 + (cc - col) ** 2))
            row, col = int(rr[k]), int(cc[k])
        world.spawns.append((row, col))
    _CACHE[world_id] = world
    return world
