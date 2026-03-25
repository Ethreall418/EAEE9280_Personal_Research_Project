"""
benchmark_parallel.py
=====================
Wall-time comparison of three execution modes for identical workloads:

  single_run         — jax.jit(run())  for a single OceanState
  batch_run          — eqx.filter_vmap over N ensemble members (1 GPU)
  sharded_ensemble   — NamedSharding across all available GPUs

Usage
-----
    python runtime_test/benchmark_parallel.py [N_ENSEMBLE] [N_STEPS]

Defaults: N_ENSEMBLE=4, N_STEPS=288 (one simulated day at dt=300 s).

Output
------
Prints a table of wall times (compilation excluded) and throughput
in member-steps/second.  On a single-device machine,
sharded_ensemble_run falls back to batch_run, so the two should match.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

# Allow running from repo root or from runtime_test/
sys.path.insert(0, str(Path(__file__).parent.parent))

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, create_rest_state
from OceanJAX.timeStepping import run as ocean_run
from OceanJAX.parallel.ensemble import batch_run, sharded_ensemble_run


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_ENSEMBLE  = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N_STEPS     = int(sys.argv[2]) if len(sys.argv) > 2 else 288
N_WARMUP    = N_STEPS           # one full run used as JIT warm-up

NX, NY, NZ  = 20, 15, 10       # matches experiment.py default
DT          = 300.0


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_grid():
    dz = 500.0 / NZ
    z  = (np.arange(NZ) + 0.5) * dz
    return OceanGrid.create(
        lon_bounds=(-40.0, -5.0),
        lat_bounds=(-15.0, 15.0),
        depth_levels=z,
        Nx=NX, Ny=NY,
    )


def timed(fn, *args, n_runs: int = 3):
    """
    Run fn(*args) n_runs times (after the first call which is used only to
    ensure compilation is done) and return (mean_wall_seconds, last_result).
    """
    # ensure compiled
    result = fn(*args)
    jax.block_until_ready(result[0].T if isinstance(result, tuple) else result.T)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result[0].T if isinstance(result, tuple) else result.T)
        times.append(time.perf_counter() - t0)

    return float(np.mean(times)), result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("OceanJAX parallel benchmark")
    print(f"  grid      : {NX}x{NY}x{NZ}  dt={DT} s")
    print(f"  steps     : {N_STEPS}  ({N_STEPS*DT/86400:.1f} sim-days)")
    print(f"  ensemble  : {N_ENSEMBLE} members")
    print(f"  devices   : {jax.devices()}")
    print("=" * 64)

    grid   = build_grid()
    params = ModelParams(dt=DT)
    base   = create_rest_state(grid, T_background=10.0, S_background=35.0)

    # Batched state: leading axis of size N_ENSEMBLE
    batched = jax.tree_util.tree_map(lambda x: jnp.stack([x] * N_ENSEMBLE), base)

    # JIT-compiled single-run function
    run_jit = jax.jit(ocean_run, static_argnames=("n_steps", "save_history"))

    # ------------------------------------------------------------------
    # [A] single_run: run each member sequentially (baseline)
    # ------------------------------------------------------------------
    print("\n[1/3] Warming up single_run ...")

    def single_sequential(n_steps):
        """Run N_ENSEMBLE members back-to-back (sequential baseline)."""
        s = base
        for _ in range(N_ENSEMBLE):
            s, _ = run_jit(s, grid, params, n_steps=n_steps, save_history=False)
        return s, None

    wall_single, _ = timed(single_sequential, N_STEPS)
    tput_single     = N_ENSEMBLE * N_STEPS / wall_single

    # ------------------------------------------------------------------
    # [B] batch_run
    # ------------------------------------------------------------------
    print("[2/3] Warming up batch_run ...")

    def run_batch(n_steps):
        return batch_run(batched, grid, params, n_steps=n_steps, save_history=False)

    wall_batch, _ = timed(run_batch, N_STEPS)
    tput_batch     = N_ENSEMBLE * N_STEPS / wall_batch

    # ------------------------------------------------------------------
    # [C] sharded_ensemble_run
    # ------------------------------------------------------------------
    print("[3/3] Warming up sharded_ensemble_run ...")

    def run_sharded(n_steps):
        return sharded_ensemble_run(batched, grid, params, n_steps=n_steps,
                                    save_history=False)

    wall_sharded, _ = timed(run_sharded, N_STEPS)
    tput_sharded     = N_ENSEMBLE * N_STEPS / wall_sharded

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    speedup_batch   = wall_single / wall_batch
    speedup_sharded = wall_single / wall_sharded

    print()
    print(f"{'Mode':<25}  {'Wall time (s)':>13}  {'member-steps/s':>15}  {'Speedup':>8}")
    print("-" * 68)
    print(f"{'single_run (sequential)':<25}  {wall_single:13.3f}  {tput_single:15.1f}  {'1.00x':>8}")
    print(f"{'batch_run (vmap)':<25}  {wall_batch:13.3f}  {tput_batch:15.1f}  {speedup_batch:>7.2f}x")
    print(f"{'sharded_ensemble_run':<25}  {wall_sharded:13.3f}  {tput_sharded:15.1f}  {speedup_sharded:>7.2f}x")
    print()

    n_devices = len(jax.devices())
    if n_devices == 1:
        print("Note: only 1 device available — sharded_ensemble_run falls back "
              "to batch_run (identical performance expected).")

    print("\nDone.")


if __name__ == "__main__":
    main()
