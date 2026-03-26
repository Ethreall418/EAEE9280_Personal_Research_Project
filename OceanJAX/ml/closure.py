"""
OceanJAX ML – Closure Interface
================================
Defines the minimal, stable ML hook that connects learnable parameterisations
to the physical time-stepping loop.

Design goals
------------
1. **Zero overhead when unused.**
   Pass ``closure=None`` (the default in every caller) to skip the hook
   entirely; the numerical path is bit-identical to the pure-physics version.

2. **Single, centralised hook in ``step()``.**
   All ML corrections enter through one place, after the explicit physics
   tendencies are computed and before the implicit vertical diffusion.
   This makes the interface easy to reason about and to test.

3. **Minimal first-version scope.**
   Only tracer tendencies (T, S) and a vertical diffusivity scaling factor
   are exposed.  Momentum corrections are intentionally deferred; they
   require extra care around the leapfrog history and the Asselin filter.

4. **Fully differentiable.**
   Both ``AbstractClosure`` and ``NullClosure`` are ``eqx.Module`` subclasses.
   Any concrete closure whose ``__call__`` is written in JAX is automatically
   compatible with ``jax.grad``, ``jax.jit``, and ``eqx.filter_vmap``.

Public API
----------
ClosureOutput       – named container for the three correction fields.
AbstractClosure     – base class; subclass and override ``__call__``.
NullClosure         – no-op implementation; preserves pure-physics behaviour.

Extension recipe
----------------
To add a real parameterisation:

    import equinox as eqx
    import jax.numpy as jnp
    from OceanJAX.ml.closure import AbstractClosure, ClosureOutput

    class MyClosure(AbstractClosure):
        # Any eqx.Module fields (weights, hyper-params, …)
        weight: jnp.ndarray

        def __call__(self, state, grid, params):
            dT = self.weight * some_function(state, grid, params)
            dS = jnp.zeros_like(dT)
            return ClosureOutput(dT_tend=dT, dS_tend=dS,
                                 kappa_v_scale=jnp.array(1.0))

    closure = MyClosure(weight=jnp.zeros((Nx, Ny, Nz)))
    final, _ = run(state, grid, params, n_steps, closure=closure)
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from OceanJAX.grid import OceanGrid
from OceanJAX.state import OceanState, ModelParams


# ---------------------------------------------------------------------------
# ClosureOutput — container for the three correction fields
# ---------------------------------------------------------------------------

class ClosureOutput(eqx.Module):
    """
    Output of a single closure evaluation.

    Fields
    ------
    dT_tend : (Nx, Ny, Nz) [K s-1]
        Additional tendency added to the explicit T tendency G_T before the
        Adams-Bashforth advance.  Land points should be zeroed by the closure
        (or will be zeroed by the mask applied afterwards).

    dS_tend : (Nx, Ny, Nz) [psu s-1]
        Same for salinity.

    kappa_v_scale : scalar or (Nx, Ny, Nz+1), dimensionless
        Multiplicative factor applied to ``params.kappa_v`` before the
        implicit vertical diffusion step.  A value of 1.0 everywhere leaves
        the background diffusivity unchanged.  Values > 1 enhance mixing
        (e.g. shear-driven or convective regions); values in (0, 1) suppress
        it.  Must be non-negative everywhere.
    """
    dT_tend:       jnp.ndarray   # (Nx, Ny, Nz)
    dS_tend:       jnp.ndarray   # (Nx, Ny, Nz)
    kappa_v_scale: jnp.ndarray   # scalar or (Nx, Ny, Nz+1)


# ---------------------------------------------------------------------------
# AbstractClosure — base class
# ---------------------------------------------------------------------------

class AbstractClosure(eqx.Module):
    """
    Base class for all ML / physics closures.

    Subclass this and override ``__call__`` to implement a concrete
    parameterisation.  All fields stored in the subclass are treated as
    equinox Module leaves and are therefore:

      - transparently handled by ``eqx.filter_vmap`` (static fields are
        replicated, array fields can be vmapped or sharded);
      - compatible with ``jax.grad`` / ``eqx.filter_grad`` for training.

    The call signature is fixed:

        def __call__(
            self,
            state  : OceanState,
            grid   : OceanGrid,
            params : ModelParams,
        ) -> ClosureOutput:
            ...
    """

    def __call__(
        self,
        state:  OceanState,
        grid:   OceanGrid,
        params: ModelParams,
    ) -> ClosureOutput:
        raise NotImplementedError(
            f"{type(self).__name__} must implement __call__."
        )


# ---------------------------------------------------------------------------
# NullClosure — no-op; preserves pure-physics behaviour
# ---------------------------------------------------------------------------

class NullClosure(AbstractClosure):
    """
    No-op closure that adds zero corrections and leaves kappa_v unchanged.

    Using ``NullClosure`` gives results numerically identical to passing
    ``closure=None``, but is useful when you want a concrete object in
    the signature (e.g., for tracing or serialisation) without activating
    any ML parameterisation.

    The zero arrays are materialised lazily from grid dimensions, so no
    large tensors are stored in the module itself.
    """

    def __call__(
        self,
        state:  OceanState,
        grid:   OceanGrid,
        params: ModelParams,
    ) -> ClosureOutput:
        zeros = jnp.zeros(
            (grid.Nx, grid.Ny, grid.Nz), dtype=jnp.float32
        )
        return ClosureOutput(
            dT_tend       = zeros,
            dS_tend       = zeros,
            kappa_v_scale = jnp.array(1.0, dtype=jnp.float32),
        )
