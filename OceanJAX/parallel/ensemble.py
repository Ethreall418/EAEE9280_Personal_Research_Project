"""
OceanJAX Parallel – Ensemble Module
=====================================
Batch / multi-GPU execution of independent OceanJAX model instances.

Public interfaces
-----------------
batch_step()           — vmap a single time step over an ensemble batch.
batch_run()            — vmap a multi-step run over an ensemble batch.
sharded_ensemble_run() — distribute the ensemble batch across multiple GPUs
                         via JAX NamedSharding + jit.

Design position
---------------
These functions distribute the *ensemble / batch dimension* across devices.
They do NOT perform domain decomposition (splitting Nx×Ny across devices).
Single-domain multi-GPU execution requires a separate shard_map / halo-
exchange layer (Phase 2/3 of the parallel roadmap).

grid and params are replicated on all devices.  The batch axis of states
(and forcing_sequence, when supplied) is the only axis that is sharded.

Composability
-------------
All three functions are compatible with jax.jit, jax.grad, and jax.vmap.
eqx.filter_vmap is used (rather than jax.vmap) to correctly handle the
static fields inside OceanGrid and ModelParams equinox modules.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from OceanJAX.grid import OceanGrid
from OceanJAX.state import OceanState, ModelParams
from OceanJAX.timeStepping import step, run, SurfaceForcing


# ---------------------------------------------------------------------------
# Public: batch_step
# ---------------------------------------------------------------------------

def batch_step(
    states:  OceanState,
    grid:    OceanGrid,
    params:  ModelParams,
    forcing: Optional[SurfaceForcing] = None,
) -> OceanState:
    """
    Advance a batch of independent ocean states by one time step.

    Uses ``eqx.filter_vmap`` to vectorise ``step()`` over the leading batch
    axis B of ``states``.  ``grid`` and ``params`` are shared across all
    ensemble members (not batched).

    Parameters
    ----------
    states  : OceanState whose array fields carry a leading batch axis B,
              e.g. ``states.T.shape == (B, Nx, Ny, Nz)``.
    grid    : OceanGrid — replicated, same for all members.
    params  : ModelParams — replicated, same for all members.
    forcing : SurfaceForcing with a leading batch axis B (same B as states),
              or None for no surface forcing.

    Returns
    -------
    OceanState with the same leading batch axis B, advanced by one dt.
    """
    if forcing is None:
        return eqx.filter_vmap(
            lambda s: step(s, grid, params, None)
        )(states)
    return eqx.filter_vmap(
        lambda s, f: step(s, grid, params, f),
        in_axes=(0, 0),
    )(states, forcing)


# ---------------------------------------------------------------------------
# Public: batch_run
# ---------------------------------------------------------------------------

def batch_run(
    states:           OceanState,
    grid:             OceanGrid,
    params:           ModelParams,
    n_steps:          int,
    forcing_sequence: Optional[SurfaceForcing] = None,
    save_history:     bool = False,
) -> tuple[OceanState, Optional[OceanState]]:
    """
    Run a batch of independent ocean simulations for ``n_steps`` steps.

    Uses ``eqx.filter_vmap`` to vectorise ``run()`` over the leading batch
    axis B of ``states``.

    Parameters
    ----------
    states           : OceanState with a leading batch axis B.
    grid             : OceanGrid — replicated.
    params           : ModelParams — replicated.
    n_steps          : Number of time steps (static — determines XLA trace).
    forcing_sequence : SurfaceForcing whose array fields have shape
                       ``(B, n_steps, Nx, Ny)``, or None.
    save_history     : If True, return all intermediate states (memory-
                       intensive for large B × n_steps).

    Returns
    -------
    (final_states, history)
      final_states : OceanState with leading batch axis B.
      history      : OceanState with axes (B, n_steps, ...) if
                     save_history=True, else None.
    """
    if forcing_sequence is None:
        return eqx.filter_vmap(
            lambda s: run(s, grid, params, n_steps, None, save_history)
        )(states)
    return eqx.filter_vmap(
        lambda s, f: run(s, grid, params, n_steps, f, save_history),
        in_axes=(0, 0),
    )(states, forcing_sequence)


# ---------------------------------------------------------------------------
# Public: sharded_ensemble_run
# ---------------------------------------------------------------------------

def sharded_ensemble_run(
    states:           OceanState,
    grid:             OceanGrid,
    params:           ModelParams,
    n_steps:          int,
    forcing_sequence: Optional[SurfaceForcing] = None,
    save_history:     bool = False,
    devices:          Optional[list] = None,
) -> tuple[OceanState, Optional[OceanState]]:
    """
    Run an ensemble batch distributed across multiple GPUs.

    The ensemble (batch) dimension is sharded across the available devices
    using ``NamedSharding(mesh, PartitionSpec('batch'))``.  ``grid`` and
    ``params`` are replicated on every device by JAX's compiler.

    Scope
    -----
    This function distributes *independent model instances* across devices.
    It does NOT split a single large domain across GPUs — that requires
    a shard_map / halo-exchange approach (Phase 2/3).

    Parameters
    ----------
    states           : OceanState with a leading batch axis B.
                       When multiple devices are used, B must be divisible
                       by the number of devices.
    grid             : OceanGrid — replicated on all devices.
    params           : ModelParams — replicated on all devices.
    n_steps          : Number of time steps (static).
    forcing_sequence : SurfaceForcing with shape ``(B, n_steps, Nx, Ny)``
                       per field, or None.
    save_history     : Passed through to batch_run.
    devices          : Devices to use.  Defaults to all available devices
                       (``jax.devices()``).  Pass a subset for partial use.

    Returns
    -------
    Same as batch_run: (final_states, history).

    Notes
    -----
    Single-device fallback: when only one device is available (or
    ``len(devices) == 1``), sharding is skipped and the function is
    equivalent to ``jax.jit(batch_run)`` — no code-path change needed.
    """
    devices = devices or jax.devices()
    n_devices = len(devices)

    # Validate batch size
    batch_size = jax.tree_util.tree_leaves(states)[0].shape[0]
    if n_devices > 1 and batch_size % n_devices != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by the number of "
            f"devices ({n_devices}).  Reduce N_ENSEMBLE or pass a device subset."
        )

    # JIT-compiled inner function — grid/params captured, not traced per call
    run_fn = jax.jit(
        lambda s, f: batch_run(s, grid, params, n_steps, f, save_history)
    )

    if n_devices == 1:
        # Single device: no sharding overhead
        return run_fn(states, forcing_sequence)

    # Multi-device: shard the batch axis across devices
    mesh             = Mesh(np.array(devices), axis_names=("batch",))
    batch_sharding   = NamedSharding(mesh, PartitionSpec("batch"))

    states_sharded   = jax.device_put(states,           batch_sharding)
    forcing_sharded  = (jax.device_put(forcing_sequence, batch_sharding)
                        if forcing_sequence is not None else None)

    return run_fn(states_sharded, forcing_sharded)
