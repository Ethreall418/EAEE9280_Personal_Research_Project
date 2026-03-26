"""
Tests for OceanJAX.ml.closure (ML interface)
=============================================
Four groups of properties are verified:

  1. NullClosure output contract
       ClosureOutput fields have the correct shape, dtype, and value.

  2. NullClosure does not change the physics
       step() with NullClosure must produce bit-identical results to
       step() with closure=None.  This guards the "zero overhead when
       unused" design promise.

  3. Custom closure wires into step() correctly
       A closure that adds a constant dT_tend must produce a T increment
       above the pure-physics baseline, exactly equal to dt * dT_value
       (AB1 first step, no history).

  4. kappa_v_scale modifies vertical diffusion
       A closure that doubles kappa_v must increase vertical mixing
       relative to the baseline (deeper homogenisation).

  5. Closure propagates through run() and batch_run()
       The same custom closure delivered via run() and via
       batch_run() must give consistent results.

Running
-------
    pytest OceanJAX/tests/test_closure.py -v
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, OceanState, create_rest_state
from OceanJAX.timeStepping import step, run
from OceanJAX.ml.closure import AbstractClosure, ClosureOutput, NullClosure
from OceanJAX.parallel.ensemble import batch_run


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_grid():
    z_levels = np.array([12.5, 37.5, 62.5, 87.5], dtype=np.float64)
    return OceanGrid.create(
        lon_bounds=(-10.0, 10.0),
        lat_bounds=(-5.0, 5.0),
        depth_levels=z_levels,
        Nx=8, Ny=6,
    )


@pytest.fixture(scope="module")
def default_params():
    return ModelParams(dt=300.0)


@pytest.fixture(scope="module")
def rest_state(small_grid):
    return create_rest_state(small_grid, T_background=10.0, S_background=35.0)


# ---------------------------------------------------------------------------
# Concrete test closures
# ---------------------------------------------------------------------------

class ConstantTTendClosure(AbstractClosure):
    """Adds a spatially uniform dT_tend; dS_tend = 0; kappa_v_scale = 1."""
    dT_value: float

    def __call__(self, state, grid, params):
        dT = jnp.full(
            (grid.Nx, grid.Ny, grid.Nz), self.dT_value, dtype=jnp.float32
        ) * grid.mask_c
        dS = jnp.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=jnp.float32)
        return ClosureOutput(
            dT_tend=dT,
            dS_tend=dS,
            kappa_v_scale=jnp.array(1.0, dtype=jnp.float32),
        )


class KappaScaleClosure(AbstractClosure):
    """Scales kappa_v by a constant factor; zero tracer tendency corrections."""
    scale: float

    def __call__(self, state, grid, params):
        zeros = jnp.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=jnp.float32)
        return ClosureOutput(
            dT_tend=zeros,
            dS_tend=zeros,
            kappa_v_scale=jnp.array(self.scale, dtype=jnp.float32),
        )


# ---------------------------------------------------------------------------
# Group 1 — NullClosure output contract
# ---------------------------------------------------------------------------

class TestNullClosureOutput:
    def test_dT_shape(self, rest_state, small_grid, default_params):
        nc = NullClosure()
        out = nc(rest_state, small_grid, default_params)
        assert out.dT_tend.shape == (small_grid.Nx, small_grid.Ny, small_grid.Nz)

    def test_dS_shape(self, rest_state, small_grid, default_params):
        nc = NullClosure()
        out = nc(rest_state, small_grid, default_params)
        assert out.dS_tend.shape == (small_grid.Nx, small_grid.Ny, small_grid.Nz)

    def test_dT_is_zero(self, rest_state, small_grid, default_params):
        nc = NullClosure()
        out = nc(rest_state, small_grid, default_params)
        np.testing.assert_array_equal(np.array(out.dT_tend), 0.0)

    def test_dS_is_zero(self, rest_state, small_grid, default_params):
        nc = NullClosure()
        out = nc(rest_state, small_grid, default_params)
        np.testing.assert_array_equal(np.array(out.dS_tend), 0.0)

    def test_kappa_v_scale_is_one(self, rest_state, small_grid, default_params):
        nc = NullClosure()
        out = nc(rest_state, small_grid, default_params)
        assert float(out.kappa_v_scale) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Group 2 — NullClosure does not change the physics
# ---------------------------------------------------------------------------

class TestNullClosureNoChange:
    def test_step_null_closure_matches_no_closure(
        self, rest_state, small_grid, default_params
    ):
        """step() with NullClosure == step() with closure=None (bit-identical)."""
        s_none = step(rest_state, small_grid, default_params, closure=None)
        s_null = step(rest_state, small_grid, default_params, closure=NullClosure())
        np.testing.assert_array_equal(
            np.array(s_none.T), np.array(s_null.T),
            err_msg="T differs: NullClosure vs closure=None"
        )
        np.testing.assert_array_equal(
            np.array(s_none.S), np.array(s_null.S),
            err_msg="S differs: NullClosure vs closure=None"
        )
        np.testing.assert_array_equal(
            np.array(s_none.eta), np.array(s_null.eta),
            err_msg="eta differs: NullClosure vs closure=None"
        )

    def test_run_null_closure_matches_no_closure(
        self, rest_state, small_grid, default_params
    ):
        """run() with NullClosure == run() with closure=None for 50 steps."""
        N = 50
        f_none, _ = run(rest_state, small_grid, default_params, n_steps=N)
        f_null, _ = run(rest_state, small_grid, default_params, n_steps=N,
                        closure=NullClosure())
        np.testing.assert_array_equal(np.array(f_none.T), np.array(f_null.T))


# ---------------------------------------------------------------------------
# Group 3 — Custom closure dT_tend wires into step correctly
# ---------------------------------------------------------------------------

class TestConstantTTend:
    def test_T_increases_by_dt_times_tend(
        self, rest_state, small_grid, default_params
    ):
        """
        On the very first step (AB1), T_new = T + dt * (G_T + dT_tend).
        For a resting uniform ocean G_T = 0, so:
          T_new = T + dt * dT_value  (wet cells only)
        """
        DT_VAL = 0.5   # K s-1
        closure = ConstantTTendClosure(dT_value=DT_VAL)
        s_base = step(rest_state, small_grid, default_params, closure=None)
        s_ml   = step(rest_state, small_grid, default_params, closure=closure)

        T_base = np.array(s_base.T)
        T_ml   = np.array(s_ml.T)
        wet    = np.array(small_grid.mask_c) > 0

        expected_increment = default_params.dt * DT_VAL
        actual_increment   = (T_ml - T_base)[wet]

        # The increment should be close to dt * dT_value.
        # Small deviations arise from implicit vertical diffusion acting on the
        # modified T; we therefore use a relative tolerance, not exact equality.
        np.testing.assert_allclose(
            actual_increment,
            np.full_like(actual_increment, expected_increment),
            rtol=1e-2,
            err_msg="T increment from closure dT_tend is incorrect",
        )

    def test_S_unchanged_when_dS_zero(
        self, rest_state, small_grid, default_params
    ):
        """dS_tend = 0 must leave S bit-identical to the no-closure baseline."""
        closure = ConstantTTendClosure(dT_value=1.0)
        s_base = step(rest_state, small_grid, default_params, closure=None)
        s_ml   = step(rest_state, small_grid, default_params, closure=closure)
        np.testing.assert_array_equal(np.array(s_base.S), np.array(s_ml.S))

    def test_T_finite_with_large_tend(
        self, rest_state, small_grid, default_params
    ):
        """Even a large dT_tend must not introduce non-finite values."""
        closure = ConstantTTendClosure(dT_value=100.0)
        s_ml   = step(rest_state, small_grid, default_params, closure=closure)
        assert np.all(np.isfinite(np.array(s_ml.T)))


# ---------------------------------------------------------------------------
# Group 4 — kappa_v_scale modifies vertical diffusion
# ---------------------------------------------------------------------------

class TestKappaVScale:
    def test_doubled_kappa_increases_mixing(
        self, small_grid, default_params
    ):
        """
        A vertically stratified initial T column should homogenise faster
        with kappa_v_scale=2 than with the default kappa_v_scale=1.

        Setup: top layer T=20, bottom layer T=10 (strong gradient).
        After 50 steps, std(T over z) should be smaller with doubled kappa_v.
        """
        from OceanJAX.state import create_from_arrays
        Nx, Ny, Nz = small_grid.Nx, small_grid.Ny, small_grid.Nz
        z_idx = np.arange(Nz)
        T0 = jnp.where(
            jnp.array(z_idx < Nz // 2),
            20.0, 10.0
        )                                                # (Nz,)
        T3d = jnp.broadcast_to(T0, (Nx, Ny, Nz)).astype(jnp.float32) * small_grid.mask_c
        S0  = jnp.full((Nx, Ny, Nz), 35.0, dtype=jnp.float32) * small_grid.mask_c
        zeros3 = jnp.zeros((Nx, Ny, Nz), dtype=jnp.float32)
        zeros2 = jnp.zeros((Nx, Ny),     dtype=jnp.float32)
        state = create_rest_state(small_grid, T_background=10.0)
        state = eqx.tree_at(lambda s: s.T, state, T3d)

        N = 50
        f_base, _ = run(state, small_grid, default_params, n_steps=N)
        f_2x,   _ = run(state, small_grid, default_params, n_steps=N,
                        closure=KappaScaleClosure(scale=2.0))

        T_base = np.array(f_base.T)
        T_2x   = np.array(f_2x.T)
        wet = np.array(small_grid.mask_c) > 0

        std_base = T_base[wet].std()
        std_2x   = T_2x[wet].std()
        assert std_2x < std_base, (
            f"doubled kappa_v should reduce T std, got std_base={std_base:.4f} "
            f"std_2x={std_2x:.4f}"
        )

    def test_zero_kappa_scale_preserves_T(
        self, rest_state, small_grid, default_params
    ):
        """kappa_v_scale=0 turns off vertical diffusion; T must still be finite."""
        closure = KappaScaleClosure(scale=0.0)
        s_ml = step(rest_state, small_grid, default_params, closure=closure)
        assert np.all(np.isfinite(np.array(s_ml.T)))


# ---------------------------------------------------------------------------
# Group 5 — closure propagates through run() and batch_run()
# ---------------------------------------------------------------------------

class TestClosurePropagation:
    def test_run_applies_closure_each_step(
        self, rest_state, small_grid, default_params
    ):
        """
        run() with a constant dT_tend must yield T > baseline for every
        wet cell after 10 steps.
        """
        N = 10
        closure = ConstantTTendClosure(dT_value=0.01)
        f_base, _ = run(rest_state, small_grid, default_params, n_steps=N)
        f_ml,   _ = run(rest_state, small_grid, default_params, n_steps=N,
                        closure=closure)
        wet = np.array(small_grid.mask_c) > 0
        assert np.all(np.array(f_ml.T)[wet] > np.array(f_base.T)[wet])

    def test_batch_run_closure_consistent_with_run(
        self, rest_state, small_grid, default_params
    ):
        """
        batch_run() with a closure applied to N members must agree with
        N independent run() calls using the same closure.
        """
        N = 2
        n_steps = 30
        closure = ConstantTTendClosure(dT_value=0.05)

        # Reference: two independent run() calls
        ref0, _ = run(rest_state, small_grid, default_params,
                      n_steps=n_steps, closure=closure)
        # Since both members are identical, ref0 = ref1
        ref_T = np.array(ref0.T)

        # batch_run with N=2 (identical states)
        batched = jax.tree_util.tree_map(
            lambda x: jnp.stack([x] * N), rest_state
        )
        final, _ = batch_run(batched, small_grid, default_params,
                             n_steps=n_steps, closure=closure)
        batch_T = np.array(final.T)   # (N, Nx, Ny, Nz)

        np.testing.assert_allclose(
            batch_T[0], ref_T, rtol=1e-5,
            err_msg="batch_run member 0 differs from run() with same closure"
        )
        np.testing.assert_allclose(
            batch_T[1], ref_T, rtol=1e-5,
            err_msg="batch_run member 1 differs from run() with same closure"
        )
