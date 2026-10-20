"""Speed comparison: optimised follower_cpp.planner vs the original implementation (tests/reference).

Run: python tests/bench_planner.py [--repeats 5]
Reports the best-of-N wall time of identical call sequences on both implementations.
"""
import argparse
import time

import numpy as np

from planner_common import (RADIUS, free_cells, maze_grid, new_planner, occupation_window, random_grid, ref_planner,
                            warehouse_like_grid)

MAPS = {
    'random 20x20 (d=0.3)': lambda: random_grid(20, 0.3, 0),
    'maze 65x65': lambda: maze_grid(65, 0),
    'random 64x64 (d=0.2)': lambda: random_grid(64, 0.2, 0),
    'warehouse 46x33': lambda: warehouse_like_grid(33, 46, 0),
}


def best_of(fn, repeats):
    best = float('inf')
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def bench_precompute(make, grid, repeats):
    g = grid.tolist()
    return best_of(lambda: make(g, True, True, True).precompute_penalty_matrix(RADIUS), repeats)


def prepared(make, grid):
    g = grid.tolist()
    p = make(g, True, True, True)
    p.set_penalties(new_planner(g, True, True, True).precompute_penalty_matrix(RADIUS))
    return p


def bench_new_goal(make, grid, repeats, n=100):
    """Every query has a new goal -> h_values recompute + A*."""
    cells = free_cells(grid)
    rng = np.random.default_rng(0)
    qs = [(cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]) for _ in range(n)]
    p = prepared(make, grid)

    def run():
        for s, g in qs:
            p.update_path(s, g)
            p.get_path()
    return best_of(run, repeats) / n


def bench_same_goal(make, grid, repeats, n=500):
    """Fixed goal, varying start -> A* only (the common per-step case)."""
    cells = free_cells(grid)
    rng = np.random.default_rng(1)
    goal = cells[rng.integers(len(cells))]
    starts = [cells[rng.integers(len(cells))] for _ in range(n)]
    p = prepared(make, grid)
    p.update_path(starts[0], goal)

    def run():
        for s in starts:
            p.update_path(s, goal)
            p.get_path()
    return best_of(run, repeats) / n


def bench_episode(make, grid, repeats, agents=16, steps=200):
    """Closest to production: per agent per step update_occupations + update_path + get_path, goals on arrival."""
    cells = free_cells(grid)
    g = grid.tolist()
    pen = new_planner(g, True, True, True).precompute_penalty_matrix(RADIUS)

    rng0 = np.random.default_rng(3)
    wins = [occupation_window(rng0, grid, (0, 0)) for _ in range(64)]  # generated outside the timed region

    def run():
        rng = np.random.default_rng(2)
        for _ in range(agents):
            p = make(g, True, True, True)
            p.set_penalties(pen)
            pos, goal = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
            for _ in range(steps):
                win, cur = wins[rng.integers(len(wins))]
                p.update_occupations(win, cur, goal)
                p.update_path(pos, goal)
                path = p.get_path()
                if len(path) > 1:
                    pos = path[1]
                if pos == goal:
                    goal = cells[rng.integers(len(cells))]
    return best_of(run, repeats) / (agents * steps)


BENCHES = [
    ('precompute_penalty_matrix (once/map)', bench_precompute, 1e3, 'ms'),
    ('update_path, new goal each call', bench_new_goal, 1e6, 'us'),
    ('update_path, same goal', bench_same_goal, 1e6, 'us'),
    ('episode step (occ+path+get_path)', bench_episode, 1e6, 'us'),
]


def bench_cache(grid, repeats, num_agents=32):
    """Cost of the per-episode-reset setup: build per-agent planners + obtain penalties (first update() call)."""
    import follower.planning as planning
    g = grid.tolist()
    cfg = planning.PlannerConfig()

    def reset(make, penalties):
        def run():
            ps = [make(g, True, True, True) for _ in range(num_agents)]
            pen = penalties()
            for p in ps:
                p.set_penalties(pen)
        return best_of(run, repeats)

    old = reset(ref_planner, lambda: ref_planner(g, True, True, True).precompute_penalty_matrix(RADIUS))
    new = reset(new_planner, lambda: new_planner(g, True, True, True).precompute_penalty_matrix(RADIUS))
    planning._PENALTY_CACHE.clear()
    planning.get_penalties(g, RADIUS, cfg)  # warm
    hit = reset(new_planner, lambda: planning.get_penalties(g, RADIUS, cfg))
    return old * 1e3, new * 1e3, hit * 1e3


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=5)
    args = ap.parse_args()
    print('\nEpisode reset (32 agents: planner construction + penalties), ms')
    print(f'{"map":24s} {"orig":>10s} {"new A*":>10s} {"new+cache":>10s} {"total speedup":>14s}')
    for map_name, mk in MAPS.items():
        old, new, hit = bench_cache(mk(), max(1, args.repeats // 2))
        print(f'{map_name:24s} {old:10.2f} {new:10.2f} {hit:10.2f} {old / hit:13.1f}x')
    print()
    print(f'{"map":24s} {"benchmark":38s} {"old":>10s} {"new":>10s} {"speedup":>8s}')
    for map_name, mk in MAPS.items():
        grid = mk()
        for name, fn, scale, unit in BENCHES:
            reps = max(1, args.repeats // 2) if fn is bench_precompute else args.repeats
            old = fn(ref_planner, grid, reps) * scale
            new = fn(new_planner, grid, reps) * scale
            print(f'{map_name:24s} {name:38s} {old:8.1f}{unit} {new:8.1f}{unit} {old / new:7.1f}x')
