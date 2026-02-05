#!/usr/bin/env python3
"""
GA optimization example (DEAP) using AGRC-Surrogate opt_api.

- Baseline CST is extracted from sc1095_full1.dat
- Bounds are +-5% around each CST coefficient (with a small absolute fallback near zero)
- Fitness: maximize L/D at a target AoA & Mach (implemented as minimize 1/(L/D))

Run:
  python examples/ga_optimize_sc1095.py --airfoil sc1095_full1.dat

Optional:
  --aoa 2.0 --mach 0.3 --ngen 30 --pop 80 --seed 1
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np

from deap import base, creator, tools, algorithms

from agrc_surrogate.opt_api import cst_from_airfoil, c81_from_cst, objective_ld_at_aoa


@dataclass
class GAConfig:
    airfoil: str
    aoa: float = 2.0
    mach: float = 0.3
    pop: int = 80
    ngen: int = 30
    cxpb: float = 0.6
    mutpb: float = 0.3
    eta_cx: float = 15.0     # SBX crossover parameter
    eta_mut: float = 20.0    # polynomial mutation parameter
    indpb: float = 0.15      # per-gene mutation probability
    seed: int = 1
    abs_floor: float = 1e-3  # fallback span for near-zero coefficients
    bound_frac: float = 0.05 # +-5%


def make_bounds(cst0: np.ndarray, frac: float, abs_floor: float) -> tuple[np.ndarray, np.ndarray]:
    """
    For each coefficient:
      span = max(|c0|*frac, abs_floor)
      lb = c0 - span
      ub = c0 + span
    """
    cst0 = np.asarray(cst0, dtype=float).ravel()
    span = np.maximum(np.abs(cst0) * frac, abs_floor)
    lb = cst0 - span
    ub = cst0 + span
    return lb, ub


def clip_inplace(ind, lb, ub):
    for i in range(len(ind)):
        if ind[i] < lb[i]:
            ind[i] = float(lb[i])
        elif ind[i] > ub[i]:
            ind[i] = float(ub[i])
    return ind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--airfoil", default="sc1095_full1.dat", help="Airfoil perimeter .dat file")
    parser.add_argument("--aoa", type=float, default=2.0)
    parser.add_argument("--mach", type=float, default=0.3)
    parser.add_argument("--pop", type=int, default=80)
    parser.add_argument("--ngen", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bound_frac", type=float, default=0.05, help="+-fraction bounds around baseline CST (0.05 = 5%)")
    parser.add_argument("--abs_floor", type=float, default=1e-3, help="Absolute fallback bound span for near-zero CST coeffs")
    args = parser.parse_args()

    cfg = GAConfig(
        airfoil=args.airfoil,
        aoa=args.aoa,
        mach=args.mach,
        pop=args.pop,
        ngen=args.ngen,
        seed=args.seed,
        bound_frac=args.bound_frac,
        abs_floor=args.abs_floor,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # -------------------------
    # Baseline CST + bounds
    # -------------------------
    cst0 = cst_from_airfoil(airfoil=cfg.airfoil)  # (20,)
    lb, ub = make_bounds(cst0, cfg.bound_frac, cfg.abs_floor)

    print("Baseline CST:")
    print("  shape:", cst0.shape)
    print("  min/max:", float(np.min(cst0)), float(np.max(cst0)))
    print("Bounds:")
    print("  min(lb)/max(ub):", float(np.min(lb)), float(np.max(ub)))

    # -------------------------
    # DEAP setup (minimize)
    # -------------------------
    # Avoid redefinition errors if rerun in notebooks
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    def init_individual():
        # Start near baseline (uniform inside bounds)
        return creator.Individual([random.uniform(lb[i], ub[i]) for i in range(len(lb))])

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness evaluation
    def evaluate(ind):
        cst = np.asarray(ind, dtype=float)
        c81 = c81_from_cst(cst)  # dict with aoa/mach/Cl/Cd/Cm
        obj = objective_ld_at_aoa(c81, aoa_target=cfg.aoa, mach_target=cfg.mach)
        return (float(obj),)

    toolbox.register("evaluate", evaluate)

    # Genetic operators
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=lb.tolist(), up=ub.tolist(), eta=cfg.eta_cx)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=lb.tolist(), up=ub.tolist(), eta=cfg.eta_mut, indpb=cfg.indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # -------------------------
    # Run GA
    # -------------------------
    pop = toolbox.population(n=cfg.pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("std", np.std)

    # Evaluate initial pop
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)

    print("\nInitial best objective:", min(ind.fitness.values[0] for ind in pop))

    for gen in range(1, cfg.ngen + 1):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=cfg.cxpb, mutpb=cfg.mutpb)

        # Ensure bounds (extra safety)
        for ind in offspring:
            clip_inplace(ind, lb, ub)

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        pop = toolbox.select(pop + offspring, k=cfg.pop)
        hof.update(pop)

        record = stats.compile(pop)
        print(f"Gen {gen:03d} | min={record['min']:.6e} avg={record['avg']:.6e} std={record['std']:.6e}")

    best = hof[0]
    best_cst = np.asarray(best, dtype=float)
    best_obj = best.fitness.values[0]

    print("\n===== BEST RESULT =====")
    print("Best objective (min 1/(L/D)):", best_obj)

    # Report L/D too
    c81_best = c81_from_cst(best_cst)
    aoa_grid = c81_best["aoa"]
    mach_grid = c81_best["mach"]
    j = int(np.argmin(np.abs(mach_grid - cfg.mach)))
    cl = np.interp(cfg.aoa, aoa_grid, c81_best["Cl"][:, j])
    cd = np.interp(cfg.aoa, aoa_grid, c81_best["Cd"][:, j])
    ld = cl / max(cd, 1e-12)
    print(f"At AoA={cfg.aoa}, Mach={cfg.mach} -> Cl={cl:.6f}, Cd={cd:.6e}, L/D={ld:.3f}")

    # Save best CST to a file (so users can reuse)
    np.savetxt("best_cst.txt", best_cst, fmt="%.10e")
    print("Saved: best_cst.txt")


if __name__ == "__main__":
    main()

