"""
Frontier detection with exactly the steps and parameters of
multi_robot_exploration/frontier_coordinator.py (_detect_frontiers and
_deduplicate), applied to a belief grid (-1 unknown, 0 free, 100 occupied).
"""
import cv2  # type: ignore[import-untyped]
import numpy as np

MIN_FRONTIER_SIZE = 15      # pixels
DEDUP_RADIUS = 1.5          # m


def detect_frontiers(belief, resolution):
    """Returns (centroids [(row, col) floats], sizes [pixels]) after dedup."""
    unknown_mask = np.where(belief == -1, 255, 0).astype(np.uint8)
    free_mask = np.where(belief == 0, 255, 0).astype(np.uint8)
    obstacle_mask = np.where(belief >= 50, 255, 0).astype(np.uint8)

    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, noise_kernel)
    obs_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_obstacles = cv2.dilate(obstacle_mask, obs_kernel, iterations=1)
    safe_free_mask = cv2.bitwise_and(free_mask, cv2.bitwise_not(dilated_obstacles))
    dilated_unknown = cv2.dilate(unknown_mask, np.ones((3, 3), np.uint8), iterations=1)
    frontier_mask = cv2.bitwise_and(safe_free_mask, dilated_unknown)

    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(frontier_mask)
    points, sizes = [], []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > MIN_FRONTIER_SIZE:
            cx, cy = centroids[i]
            points.append(np.array([cy, cx]))
            sizes.append(int(stats[i, cv2.CC_STAT_AREA]))

    # Dedup (coordinator keeps the first of any cluster within DEDUP_RADIUS)
    radius = DEDUP_RADIUS / resolution
    keep_p, keep_s = [], []
    for p, s in zip(points, sizes):
        if all(np.linalg.norm(p - q) > radius for q in keep_p):
            keep_p.append(p)
            keep_s.append(s)
    return keep_p, keep_s
