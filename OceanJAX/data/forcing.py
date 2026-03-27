"""
OceanJAX data — Forcing sequence utilities
==========================================
Provides two public functions for constructing time-varying SurfaceForcing
sequences that can be passed directly to ``timeStepping.run()``.

make_forcing_sequence
---------------------
    Temporal interpolation from a sparse set of (time, SurfaceForcing)
    snapshots onto every model time step.  Three interpolation modes:

      "linear"  — piecewise-linear interpolation between snapshots.
      "nearest" — step function; each step uses the nearest snapshot.
      "cyclic"  — like "linear" but wraps the snapshot times periodically,
                  so a single annual cycle can drive an arbitrarily long run.

    Typical workflow (real ORAS5 monthly forcing):

        from OceanJAX.data.oras5 import read_oras5_forcing, regrid_forcing
        from OceanJAX.data.forcing import make_forcing_sequence

        snapshots = []
        for month_idx, t_s in enumerate(month_start_times_seconds):
            raw = read_oras5_forcing(files, time_index=month_idx)
            sf  = regrid_forcing(raw, grid)          # shape (Nx, Ny) per field
            snapshots.append((t_s, sf))

        forcing_seq = make_forcing_sequence(snapshots, n_steps, dt,
                                            interp="cyclic")
        final, _ = run(state, grid, params, n_steps,
                       forcing_sequence=forcing_seq)

make_synthetic_forcing
----------------------
    Constructs an analytical forcing sequence.  Each field can be:

      float            — spatially and temporally constant.
      np.ndarray (Nx,Ny) — spatially varying, temporally constant.
      dict             — sinusoidal in time, with optional spatial patterns:
                         {
                           "mean"      : float or (Nx, Ny),   default 0
                           "amplitude" : float or (Nx, Ny),   default 0
                           "period"    : float [s],            default 365 days
                           "phase"     : float [rad],          default 0
                         }
                         signal(t) = mean + amplitude * sin(2π t / period + phase)

    Typical workflow (seasonal cycle experiment):

        from OceanJAX.data.forcing import make_synthetic_forcing

        forcing_seq = make_synthetic_forcing(
            grid, n_steps, dt,
            heat_flux={"mean": -50.0, "amplitude": 150.0,
                       "period": 365*86400},
            tau_x=-0.05,
        )
        final, _ = run(state, grid, params, n_steps,
                       forcing_sequence=forcing_seq)
"""

from __future__ import annotations

from typing import Union

import numpy as np

from OceanJAX.grid import OceanGrid

# Type alias for a single field specification
FieldSpec = Union[float, np.ndarray, dict]

_DAYS = 86400.0
_YEAR = 365.0 * _DAYS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_spatial(spec_value, Nx: int, Ny: int, name: str) -> np.ndarray:
    """
    Return a (Nx, Ny) float32 array from a scalar or array field value.
    Raises ValueError for wrong shapes.
    """
    arr = np.asarray(spec_value, dtype=np.float32)
    if arr.ndim == 0:
        return np.full((Nx, Ny), float(arr), dtype=np.float32)
    if arr.shape == (Nx, Ny):
        return arr.astype(np.float32)
    raise ValueError(
        f"Spatial field '{name}' must be scalar or shape ({Nx}, {Ny}); "
        f"got {arr.shape}."
    )


def _build_sinusoidal(
    spec: dict,
    Nx: int,
    Ny: int,
    step_times: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Return (n_steps, Nx, Ny) float32 array for a sinusoidal field spec.

    spec keys: mean, amplitude, period, phase  (all optional)
    """
    mean_2d = _resolve_spatial(spec.get("mean",      0.0), Nx, Ny, f"{name}.mean")
    amp_2d  = _resolve_spatial(spec.get("amplitude", 0.0), Nx, Ny, f"{name}.amplitude")
    period  = float(spec.get("period", _YEAR))
    phase   = float(spec.get("phase",  0.0))

    if period <= 0:
        raise ValueError(f"Field '{name}': period must be positive, got {period}.")

    # (n_steps,) time factor
    time_factor = np.sin(2.0 * np.pi * step_times / period + phase).astype(np.float32)

    # Broadcast: (n_steps, Nx, Ny) = (n_steps, 1, 1) * (Nx, Ny)
    result = mean_2d[np.newaxis, :, :] + amp_2d[np.newaxis, :, :] * time_factor[:, np.newaxis, np.newaxis]
    return result.astype(np.float32)


def _resolve_field_to_sequence(
    spec: FieldSpec,
    Nx: int,
    Ny: int,
    step_times: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Dispatch a FieldSpec → (n_steps, Nx, Ny) float32 array.
    """
    n_steps = len(step_times)

    if isinstance(spec, dict):
        return _build_sinusoidal(spec, Nx, Ny, step_times, name)

    # Scalar or array → constant in time
    spatial = _resolve_spatial(spec, Nx, Ny, name)
    return np.broadcast_to(spatial[np.newaxis], (n_steps, Nx, Ny)).astype(np.float32).copy()


def _interp_field_stack(
    field_stack:  np.ndarray,
    src_times:    np.ndarray,
    tgt_times:    np.ndarray,
    interp:       str,
) -> np.ndarray:
    """
    Interpolate (n_slices, Nx, Ny) → (n_steps, Nx, Ny).

    Modes
    -----
    "linear"  : np.interp (piecewise linear, clamps at edges)
    "nearest" : argmin distance to source times
    "cyclic"  : linear with modulo wrap; period = src_times[-1] - src_times[0]
    """
    n_slices, Nx, Ny = field_stack.shape
    n_steps = len(tgt_times)

    if interp == "nearest":
        # (n_slices, n_steps) distance matrix → argmin over slices
        dists   = np.abs(src_times[:, np.newaxis] - tgt_times[np.newaxis, :])  # (n_slices, n_steps)
        indices = np.argmin(dists, axis=0)                                       # (n_steps,)
        return field_stack[indices].astype(np.float32)                           # (n_steps, Nx, Ny)

    if interp == "cyclic":
        if n_slices < 2:
            raise ValueError("cyclic interpolation requires at least 2 snapshots.")
        period   = src_times[-1] - src_times[0]
        t0       = src_times[0]
        # Wrap target times into [t0, t0 + period)
        tgt_wrapped = t0 + (tgt_times - t0) % period
        query_times = tgt_wrapped
    else:
        # linear (default)
        query_times = tgt_times

    # Vectorised linear interpolation: reshape to (n_slices, Nx*Ny), interp, reshape back
    flat    = field_stack.reshape(n_slices, -1).astype(np.float64)        # (n_slices, Nx*Ny)
    n_pts   = flat.shape[1]
    result  = np.empty((n_steps, n_pts), dtype=np.float64)
    for k in range(n_pts):
        result[:, k] = np.interp(query_times, src_times, flat[:, k])

    return result.reshape(n_steps, Nx, Ny).astype(np.float32)


# ---------------------------------------------------------------------------
# Public: make_forcing_sequence
# ---------------------------------------------------------------------------

def make_forcing_sequence(
    snapshots:  list[tuple[float, "SurfaceForcing"]],
    n_steps:    int,
    dt:         float,
    interp:     str   = "linear",
    t_start:    float = 0.0,
) -> "SurfaceForcing":
    """
    Build a ``(n_steps, Nx, Ny)`` SurfaceForcing by temporal interpolation
    from a sparse list of (time, SurfaceForcing) snapshots.

    Parameters
    ----------
    snapshots : list of (time_s, SurfaceForcing) tuples
        Each element is a pair ``(t, sf)`` where ``t`` is the snapshot time
        in seconds relative to the model epoch, and ``sf`` is a
        ``SurfaceForcing`` with per-field shape ``(Nx, Ny)``.
        Must be sorted by ascending time, with at least 2 entries.
    n_steps   : number of model time steps in the output sequence.
    dt        : model time step in seconds.
    interp    : interpolation mode — "linear", "nearest", or "cyclic".
    t_start   : model start time in seconds (default 0).  The first model
                step corresponds to ``t_start + dt``; all steps are offset
                by this value before interpolating.

    Returns
    -------
    SurfaceForcing with each field of shape ``(n_steps, Nx, Ny)``, ready
    to be passed as ``forcing_sequence`` to ``timeStepping.run()``.

    Notes
    -----
    - "cyclic" wraps the run time modulo the period defined by the first
      and last snapshot times.  Use this to repeat an annual cycle.
    - Fields not set in the snapshots (value == 0 everywhere) are passed
      through as zero arrays.
    """
    from OceanJAX.timeStepping import SurfaceForcing
    import jax.numpy as jnp

    if len(snapshots) < 2:
        raise ValueError("make_forcing_sequence requires at least 2 snapshots.")
    if interp not in ("linear", "nearest", "cyclic"):
        raise ValueError(f"interp must be 'linear', 'nearest', or 'cyclic'; got '{interp}'.")

    # Sort by time
    snapshots = sorted(snapshots, key=lambda x: x[0])
    src_times = np.array([t for t, _ in snapshots], dtype=np.float64)

    # Model step times (mid-step: t_start + (i+1)*dt for i in range(n_steps))
    tgt_times = t_start + (np.arange(n_steps, dtype=np.float64) + 1.0) * dt

    # Extract (n_slices, Nx, Ny) stacks for each field
    hf_stack = np.stack([np.array(sf.heat_flux) for _, sf in snapshots])
    fw_stack = np.stack([np.array(sf.fw_flux)   for _, sf in snapshots])
    tx_stack = np.stack([np.array(sf.tau_x)     for _, sf in snapshots])
    ty_stack = np.stack([np.array(sf.tau_y)     for _, sf in snapshots])

    hf_seq = _interp_field_stack(hf_stack, src_times, tgt_times, interp)
    fw_seq = _interp_field_stack(fw_stack, src_times, tgt_times, interp)
    tx_seq = _interp_field_stack(tx_stack, src_times, tgt_times, interp)
    ty_seq = _interp_field_stack(ty_stack, src_times, tgt_times, interp)

    return SurfaceForcing(
        heat_flux = jnp.asarray(hf_seq),
        fw_flux   = jnp.asarray(fw_seq),
        tau_x     = jnp.asarray(tx_seq),
        tau_y     = jnp.asarray(ty_seq),
    )


# ---------------------------------------------------------------------------
# Public: make_synthetic_forcing
# ---------------------------------------------------------------------------

def make_synthetic_forcing(
    grid:      OceanGrid,
    n_steps:   int,
    dt:        float,
    heat_flux: FieldSpec = 0.0,
    fw_flux:   FieldSpec = 0.0,
    tau_x:     FieldSpec = 0.0,
    tau_y:     FieldSpec = 0.0,
    t_start:   float = 0.0,
) -> "SurfaceForcing":
    """
    Construct an analytical ``(n_steps, Nx, Ny)`` SurfaceForcing sequence.

    Each field parameter accepts one of three forms:

    float
        Spatially and temporally constant.  E.g. ``heat_flux=-50.0``.

    np.ndarray of shape ``(Nx, Ny)``
        Spatially varying, temporally constant.  E.g. a pre-computed
        wind stress pattern.

    dict
        Sinusoidal time variation with optional spatial patterns::

            {
              "mean"      : float or (Nx, Ny),   # time-mean value
              "amplitude" : float or (Nx, Ny),   # half-amplitude of oscillation
              "period"    : float,                # period in seconds
              "phase"     : float,                # phase offset in radians
            }

        All dict keys are optional (defaults: mean=0, amplitude=0,
        period=365 days, phase=0).  The signal at step i is::

            t_i = t_start + i * dt
            f(t_i) = mean + amplitude * sin(2π t_i / period + phase)

    Parameters
    ----------
    grid     : OceanGrid (provides Nx, Ny)
    n_steps  : number of time steps in the output sequence
    dt       : model time step in seconds
    heat_flux, fw_flux, tau_x, tau_y : FieldSpec for each surface field
    t_start  : model start time in seconds (default 0)

    Returns
    -------
    SurfaceForcing with each field of shape ``(n_steps, Nx, Ny)``.
    """
    from OceanJAX.timeStepping import SurfaceForcing
    import jax.numpy as jnp

    Nx, Ny = grid.Nx, grid.Ny
    step_times = t_start + np.arange(n_steps, dtype=np.float64) * dt

    hf_seq = _resolve_field_to_sequence(heat_flux, Nx, Ny, step_times, "heat_flux")
    fw_seq = _resolve_field_to_sequence(fw_flux,   Nx, Ny, step_times, "fw_flux")
    tx_seq = _resolve_field_to_sequence(tau_x,     Nx, Ny, step_times, "tau_x")
    ty_seq = _resolve_field_to_sequence(tau_y,     Nx, Ny, step_times, "tau_y")

    return SurfaceForcing(
        heat_flux = jnp.asarray(hf_seq),
        fw_flux   = jnp.asarray(fw_seq),
        tau_x     = jnp.asarray(tx_seq),
        tau_y     = jnp.asarray(ty_seq),
    )
