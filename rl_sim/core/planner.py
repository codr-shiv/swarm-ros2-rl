"""8-connected A* on the inflated ground-truth map (stands in for Nav2's planner)."""
import heapq
import math

import numpy as np

_MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def astar(free, start, goal):
    """Path [(row, col), ...] from start to goal through *free* cells, or None."""
    if start == goal:
        return [start]
    h, w = free.shape
    gr, gc = goal

    def heur(r, c):
        dr, dc = abs(r - gr), abs(c - gc)
        return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)

    g = {start: 0.0}
    parent = {}
    openq = [(heur(*start), 0.0, start)]
    closed = set()
    while openq:
        _, cost, node = heapq.heappop(openq)
        if node in closed:
            continue
        if node == goal:
            path = [node]
            while node in parent:
                node = parent[node]
                path.append(node)
            return path[::-1]
        closed.add(node)
        r, c = node
        for dr, dc, step in _MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and free[nr, nc]:
                nxt = (nr, nc)
                ncost = cost + step
                if ncost < g.get(nxt, np.inf):
                    g[nxt] = ncost
                    parent[nxt] = node
                    heapq.heappush(openq, (ncost + heur(nr, nc), ncost, nxt))
    return None
