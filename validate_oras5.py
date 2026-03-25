"""
validate_oras5.py
=================
Two-phase validation of the ORAS5 loading pipeline and model stability.

Phase 1 — Load & regrid diagnostics
  • read_oras5() on the real merged NEMO/ORCA file
  • regrid_to_model() onto a North-Atlantic subdomain
  • Per-field statistics: shape, range, mean, NaN count

Phase 2 — Short-integration stability
  • 24-hour run (96 steps, dt = 900 s) from the ORAS5 initial state
  • Per-chunk diagnostics (T/S/eta range, non-finite flag)
  • Output saved to validate_oras5_output.nc

Usage
-----
    python validate_oras5.py
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Paths & domain
# ---------------------------------------------------------------------------

ORAS5_FILE = Path(
    r"OceanJAX/data/data_oras5/oras5_2026_01_native_merged.nc"
)

# Tropical Atlantic open-ocean subdomain (few land cells, warm pool)
LON_MIN, LON_MAX =  -40.0,   -5.0   # degrees east
LAT_MIN, LAT_MAX =  -15.0,   15.0   # degrees north
DEPTH_MAX        = 500.0             # m

# Model grid resolution
NX, NY, NZ = 20, 15, 10

# Integration parameters
# dt=300s keeps barotropic CFL = sqrt(g*DEPTH_MAX)*2*dt/dx << 1
DT         = 300.0   # s
N_STEPS    = 288     # 24 h  (288 × 300 s = 86400 s)
CHUNK_SIZE = 72      # 6-h chunks
OUTPUT_NC  = "validate_oras5_output.nc"


# ---------------------------------------------------------------------------
# Helper: print field statistics
# ---------------------------------------------------------------------------

def _field_stats(name: str, arr, mask=None) -> None:
    """Print min/max/mean and NaN count for a numpy array."""
    a = np.asarray(arr, dtype=np.float64)
    n_nan = int(np.sum(np.isnan(a)))
    valid = a[~np.isnan(a)]
    if len(valid) == 0:
        print(f"  {name:8s}: shape={a.shape}  ALL NaN")
        return

    # If a mask is provided, also report stats restricted to wet cells.
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        wet = a[m & ~np.isnan(a)]
        print(
            f"  {name:8s}: shape={a.shape}  "
            f"min={valid.min():.4g}  max={valid.max():.4g}  "
            f"mean={valid.mean():.4g}  NaN={n_nan}  "
            f"(wet: min={wet.min() if len(wet) else float('nan'):.4g} "
            f"max={wet.max() if len(wet) else float('nan'):.4g})"
        )
    else:
        print(
            f"  {name:8s}: shape={a.shape}  "
            f"min={valid.min():.4g}  max={valid.max():.4g}  "
            f"mean={valid.mean():.4g}  NaN={n_nan}"
        )


# ---------------------------------------------------------------------------
# Phase 1 — Load & regrid
# ---------------------------------------------------------------------------

def phase1():
    print("=" * 60)
    print("PHASE 1: Load & regrid diagnostics")
    print("=" * 60)

    # --- read raw ORAS5 ---
    from OceanJAX.data.oras5 import read_oras5, regrid_to_model
    from OceanJAX.grid import OceanGrid

    print(f"\nReading {ORAS5_FILE} ...")
    t0 = _time.perf_counter()
    raw = read_oras5(ORAS5_FILE, time_index=0)
    print(f"  done in {_time.perf_counter() - t0:.1f} s\n")

    print("Raw ORAS5 fields:")
    src_lon = raw["lon"]
    src_lat = raw["lat"]
    curvilinear = src_lat.ndim == 2
    print(f"  grid type : {'curvilinear (NEMO/ORCA)' if curvilinear else 'regular 1-D'}")
    print(f"  lon shape : {src_lon.shape}  "
          f"range [{np.nanmin(src_lon):.2f}, {np.nanmax(src_lon):.2f}]")
    print(f"  lat shape : {src_lat.shape}  "
          f"range [{np.nanmin(src_lat):.2f}, {np.nanmax(src_lat):.2f}]")
    print(f"  depth     : {raw['depth'].shape}  "
          f"[{raw['depth'][0]:.1f} … {raw['depth'][-1]:.1f}] m")
    _field_stats("T",   raw["T"])
    _field_stats("S",   raw["S"])
    _field_stats("u",   raw["u"])
    _field_stats("v",   raw["v"])
    _field_stats("eta", raw["eta"])

    # --- build model grid ---
    print(f"\nBuilding model grid: "
          f"lon=[{LON_MIN},{LON_MAX}]  lat=[{LAT_MIN},{LAT_MAX}]  "
          f"Nx={NX} Ny={NY} Nz={NZ}  depth_max={DEPTH_MAX} m")
    dz           = DEPTH_MAX / NZ
    depth_levels = (np.arange(NZ) + 0.5) * dz
    grid = OceanGrid.create(
        lon_bounds   = (LON_MIN, LON_MAX),
        lat_bounds   = (LAT_MIN, LAT_MAX),
        depth_levels = depth_levels,
        Nx           = NX,
        Ny           = NY,
    )
    print(f"  grid.z_c : {np.round(np.array(grid.z_c), 1)}")

    # --- regrid ---
    print(f"\nRegridding to model grid ...")
    t0 = _time.perf_counter()
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state = regrid_to_model(raw, grid)
    elapsed = _time.perf_counter() - t0
    if caught:
        for w in caught:
            print(f"  [WARNING] {w.message}")
    print(f"  done in {elapsed:.1f} s\n")

    mask3  = np.asarray(grid.mask_c) > 0   # (Nx, Ny, Nz) bool
    mask2  = mask3[:, :, 0]                # (Nx, Ny) — surface mask for eta

    print("Regridded OceanState fields:")
    _field_stats("T",   np.array(state.T),   mask3)
    _field_stats("S",   np.array(state.S),   mask3)
    _field_stats("u",   np.array(state.u),   mask3)
    _field_stats("v",   np.array(state.v),   mask3)
    _field_stats("eta", np.array(state.eta), mask2)

    # Quick sanity checks
    T_arr = np.array(state.T)
    S_arr = np.array(state.S)
    wet   = mask3

    T_wet = T_arr[wet]
    S_wet = S_arr[wet]
    ok_T  = np.all((T_wet > -2.0) & (T_wet < 35.0))
    ok_S  = np.all((S_wet > 20.0) & (S_wet < 42.0))
    ok_nan= not np.any(np.isnan(T_arr[wet])) and not np.any(np.isnan(S_arr[wet]))

    print(f"\nSanity checks:")
    print(f"  T in [-2, 35] °C  : {'PASS' if ok_T  else 'FAIL'}")
    print(f"  S in [20, 42] psu : {'PASS' if ok_S  else 'FAIL'}")
    print(f"  no NaN in wet cells: {'PASS' if ok_nan else 'FAIL'}")

    return grid, state


# ---------------------------------------------------------------------------
# Phase 1b — Tendency diagnostic (one step, no advance)
# ---------------------------------------------------------------------------

def phase1b_diag(grid, state):
    """Compute and print the momentum/tracer tendency magnitudes at t=0."""
    print("\n--- Initial tendency diagnostics ---")
    import jax.numpy as jnp
    from OceanJAX.state import ModelParams
    from OceanJAX.Physics.dynamics import (
        equation_of_state, hydrostatic_pressure,
        momentum_tendency_u, momentum_tendency_v,
    )
    from OceanJAX.Physics.tracers import tracer_tendency

    params = ModelParams(dt=DT)
    rho         = equation_of_state(state.T, state.S, params)
    p_hyd_prime = hydrostatic_pressure(rho, grid, params)

    G_u = np.array(momentum_tendency_u(state, p_hyd_prime, grid, params))
    G_v = np.array(momentum_tendency_v(state, p_hyd_prime, grid, params))
    G_T = np.array(tracer_tendency(state.T, state.u, state.v, state.w,
                                   params.kappa_h, grid))

    def _mag(name, arr):
        wet = np.asarray(grid.mask_c) > 0 if arr.shape == (NX, NY, NZ) else None
        a   = arr[wet] if wet is not None else arr.ravel()
        print(f"  |{name}|_max = {np.abs(a).max():.4g}   "
              f"rms = {np.sqrt(np.mean(a**2)):.4g}")
        # Estimate velocity change per step: G * dt
        print(f"    -> delta per dt ({DT}s): {np.abs(a).max() * DT:.4g}")

    _mag("G_u [m/s²]", G_u)
    _mag("G_v [m/s²]", G_v)
    _mag("G_T [°C/s]", G_T)


# ---------------------------------------------------------------------------
# Phase 2 helper — single-step NaN hunt (first N steps, no lax.scan)
# ---------------------------------------------------------------------------

def phase2_stepwise(grid, state, max_steps: int = 30):
    """Run step() one call at a time; stop at first NaN and report which field."""
    from OceanJAX.state import ModelParams
    from OceanJAX.timeStepping import step as ocean_step

    params   = ModelParams(dt=DT)
    step_jit = jax.jit(ocean_step)

    for n in range(1, max_steps + 1):
        state = step_jit(state, grid, params)
        jax.block_until_ready(state.T)

        fields = {
            "T":   np.array(state.T),
            "S":   np.array(state.S),
            "u":   np.array(state.u),
            "v":   np.array(state.v),
            "eta": np.array(state.eta),
        }
        bad = {k: v for k, v in fields.items() if not np.all(np.isfinite(v))}

        line = (f"  step {n:3d}  "
                f"T=[{fields['T'].min():.3f},{fields['T'].max():.3f}]  "
                f"S=[{fields['S'].min():.3f},{fields['S'].max():.3f}]  "
                f"u=[{fields['u'].min():.4f},{fields['u'].max():.4f}]  "
                f"eta=[{fields['eta'].min():.4f},{fields['eta'].max():.4f}]")
        print(line)

        if bad:
            print(f"  *** NaN/Inf first appeared in: {list(bad.keys())} ***")
            return False

    return True


# ---------------------------------------------------------------------------
# Phase 2 — Short integration
# ---------------------------------------------------------------------------

def phase2(grid, state, zero_velocity: bool = False):
    print("\n" + "=" * 60)
    print("PHASE 2: Short-integration stability (24 h)")
    print("=" * 60)

    from OceanJAX.state import ModelParams, create_from_arrays
    from OceanJAX.timeStepping import run as ocean_run
    import netCDF4 as nc_lib

    if zero_velocity:
        print("  [baroclinic cold start: u=v=0, eta=0, T/S from ORAS5]")
        zeros3 = jnp.zeros((NX, NY, NZ), dtype=jnp.float32)
        zeros2 = jnp.zeros((NX, NY),     dtype=jnp.float32)
        state  = create_from_arrays(grid, u=zeros3, v=zeros3,
                                    T=state.T, S=state.S, eta=zeros2)

    params   = ModelParams(dt=DT)
    run_jit  = jax.jit(ocean_run, static_argnames=("n_steps", "save_history"))

    # Save t = 0
    ds = _create_nc(OUTPUT_NC, grid)
    _write_snapshot(ds, state, 0)
    print(f"  Output: {OUTPUT_NC}  (t=0 saved)")

    steps_done = 0
    all_ok     = True

    try:
        while steps_done < N_STEPS:
            chunk = min(CHUNK_SIZE, N_STEPS - steps_done)
            t0 = _time.perf_counter()
            state, _ = run_jit(
                state, grid, params,
                n_steps=chunk,
                forcing_sequence=None,
                save_history=False,
            )
            jax.block_until_ready(state.T)
            wall = _time.perf_counter() - t0

            steps_done += chunk
            sim_h = float(state.time) / 3600.0

            T_a   = np.array(state.T)
            S_a   = np.array(state.S)
            eta_a = np.array(state.eta)
            bad   = (not np.all(np.isfinite(T_a)) or
                     not np.all(np.isfinite(S_a)) or
                     not np.all(np.isfinite(eta_a)))

            print(
                f"  step {steps_done:3d}/{N_STEPS}  "
                f"t={sim_h:.1f} h  "
                f"T=[{T_a.min():.3f}, {T_a.max():.3f}]  "
                f"S=[{S_a.min():.3f}, {S_a.max():.3f}]  "
                f"eta=[{eta_a.min():.4f}, {eta_a.max():.4f}]  "
                f"{'NON-FINITE!' if bad else 'ok'}  "
                f"wall={wall:.2f}s"
            )

            _write_snapshot(ds, state, steps_done)

            if bad:
                print("\nERROR: non-finite values detected — aborting.", file=sys.stderr)
                all_ok = False
                break
    finally:
        ds.close()

    print(f"\n{'PHASE 2 PASS' if all_ok else 'PHASE 2 FAIL'} — "
          f"{steps_done} steps, output in {OUTPUT_NC}")
    return all_ok


# ---------------------------------------------------------------------------
# NetCDF helpers
# ---------------------------------------------------------------------------

def _create_nc(path: str, grid) -> "netCDF4.Dataset":
    import netCDF4 as nc_lib
    ds = nc_lib.Dataset(path, mode="w", format="NETCDF4")
    ds.description = "OceanJAX ORAS5 validation run"
    ds.domain      = f"lon=[{LON_MIN},{LON_MAX}] lat=[{LAT_MIN},{LAT_MAX}]"
    ds.dt          = DT

    ds.createDimension("time", None)
    ds.createDimension("x",    grid.Nx)
    ds.createDimension("y",    grid.Ny)
    ds.createDimension("z",    grid.Nz)

    v = ds.createVariable("time", "f4", ("time",))
    v.units = "s"

    v = ds.createVariable("x", "f4", ("x",))
    v.units = "degrees_east";  v[:] = np.array(grid.lon_c)

    v = ds.createVariable("y", "f4", ("y",))
    v.units = "degrees_north"; v[:] = np.array(grid.lat_c)

    v = ds.createVariable("z", "f4", ("z",))
    v.units = "m";             v[:] = np.array(grid.z_c)

    ds.createVariable("T",   "f4", ("time", "x", "y", "z"),
                      fill_value=np.float32(np.nan))
    ds.createVariable("S",   "f4", ("time", "x", "y", "z"),
                      fill_value=np.float32(np.nan))
    ds.createVariable("eta", "f4", ("time", "x", "y"),
                      fill_value=np.float32(np.nan))
    return ds


def _write_snapshot(ds, state, step: int) -> None:
    i = len(ds.variables["time"])
    ds.variables["time"][i]        = float(state.time)
    ds.variables["T"][i, :, :, :]  = np.array(state.T)
    ds.variables["S"][i, :, :, :]  = np.array(state.S)
    ds.variables["eta"][i, :, :]   = np.array(state.eta)
    ds.sync()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not ORAS5_FILE.exists():
        print(f"ERROR: ORAS5 file not found: {ORAS5_FILE}", file=sys.stderr)
        sys.exit(1)

    grid, state = phase1()
    phase1b_diag(grid, state)

    # First try: full ORAS5 initial state (u, v from data)
    print("\n--- Phase 2a: full ORAS5 init (u,v,eta from data) ---")
    ok_full = phase2(grid, state, zero_velocity=False)

    print("\n--- Phase 2b: baroclinic cold start (u=v=0, eta=0, T/S from ORAS5) ---")
    ok_cold = phase2(grid, state, zero_velocity=True)

    print(f"\nPhase 2a (full ORAS5): {'PASS' if ok_full else 'FAIL'}")
    print(f"Phase 2b (T/S only  ): {'PASS' if ok_cold else 'FAIL'}")
    sys.exit(0 if ok_cold else 1)
