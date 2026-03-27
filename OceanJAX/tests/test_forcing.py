"""
Tests for OceanJAX.data.forcing
================================
Covers:
  1. make_synthetic_forcing — scalar constant
  2. make_synthetic_forcing — spatial (Nx, Ny) constant
  3. make_synthetic_forcing — sinusoidal dict (scalar mean/amp)
  4. make_synthetic_forcing — sinusoidal dict with spatial mean
  5. make_synthetic_forcing — multiple fields, mixed specs
  6. make_forcing_sequence  — linear interpolation, correctness at nodes and midpoints
  7. make_forcing_sequence  — nearest interpolation
  8. make_forcing_sequence  — cyclic interpolation (wraps correctly)
  9. make_forcing_sequence  — output shape
 10. Integration: make_synthetic_forcing output passes through timeStepping.run()
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp
from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, create_rest_state
from OceanJAX.timeStepping import SurfaceForcing, run
from OceanJAX.data.forcing import make_synthetic_forcing, make_forcing_sequence

_YEAR = 365.0 * 86400.0


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_grid():
    return OceanGrid.create(
        lon_bounds=(-10.0, 10.0),
        lat_bounds=(-5.0, 5.0),
        depth_levels=np.array([25.0, 75.0, 150.0], dtype=np.float64),
        Nx=4, Ny=3,
    )


@pytest.fixture(scope="module")
def rest_state(small_grid):
    return create_rest_state(small_grid)


@pytest.fixture(scope="module")
def default_params():
    return ModelParams(dt=300.0)


# ---------------------------------------------------------------------------
# Group 1 — make_synthetic_forcing
# ---------------------------------------------------------------------------

class TestMakeSyntheticForcing:

    def test_scalar_constant_shape(self, small_grid):
        """Scalar spec → (n_steps, Nx, Ny) uniform array."""
        seq = make_synthetic_forcing(small_grid, n_steps=10, dt=300.0,
                                     heat_flux=-50.0)
        assert seq.heat_flux.shape == (10, 4, 3)
        assert float(jnp.min(seq.heat_flux)) == pytest.approx(-50.0)
        assert float(jnp.max(seq.heat_flux)) == pytest.approx(-50.0)

    def test_scalar_zero_defaults(self, small_grid):
        """Default (all zeros) → all fields are zero."""
        seq = make_synthetic_forcing(small_grid, n_steps=5, dt=300.0)
        for field in (seq.heat_flux, seq.fw_flux, seq.tau_x, seq.tau_y):
            assert float(jnp.max(jnp.abs(field))) == pytest.approx(0.0)

    def test_spatial_array_constant(self, small_grid):
        """(Nx, Ny) array spec → constant in time, spatially varying."""
        Nx, Ny = small_grid.Nx, small_grid.Ny
        pattern = np.random.default_rng(0).standard_normal((Nx, Ny)).astype(np.float32)
        seq = make_synthetic_forcing(small_grid, n_steps=8, dt=300.0,
                                     tau_x=pattern)
        # All time steps should equal the pattern
        arr = np.array(seq.tau_x)
        np.testing.assert_allclose(arr[0], pattern, rtol=1e-5)
        np.testing.assert_allclose(arr[-1], pattern, rtol=1e-5)
        # Check no temporal variation
        assert np.max(np.abs(arr - arr[0:1])) == pytest.approx(0.0, abs=1e-6)

    def test_sinusoidal_mean_only(self, small_grid):
        """Dict with mean only → constant at that mean (amplitude=0)."""
        seq = make_synthetic_forcing(small_grid, n_steps=20, dt=86400.0,
                                     heat_flux={"mean": -30.0})
        arr = np.array(seq.heat_flux)
        np.testing.assert_allclose(arr, -30.0, rtol=1e-5)

    def test_sinusoidal_full_cycle(self, small_grid):
        """sin over exactly one period → starts and ends at mean."""
        n_steps = 365
        dt = 86400.0  # daily steps, 1-year run
        seq = make_synthetic_forcing(
            small_grid, n_steps=n_steps, dt=dt,
            heat_flux={"mean": 0.0, "amplitude": 100.0,
                       "period": _YEAR, "phase": 0.0},
        )
        arr = np.array(seq.heat_flux)
        # Step 0: t=0 → sin(0)=0 (mean)
        np.testing.assert_allclose(arr[0], 0.0, atol=1.0)
        # Peak near t = T/4
        peak_idx = n_steps // 4
        assert float(arr[peak_idx, 0, 0]) > 90.0

    def test_sinusoidal_spatial_mean(self, small_grid):
        """Spatial mean array: time-series at each cell offset by that mean."""
        Nx, Ny = small_grid.Nx, small_grid.Ny
        mean_field = np.ones((Nx, Ny), dtype=np.float32) * 20.0
        mean_field[0, 0] = 50.0  # one cell different
        seq = make_synthetic_forcing(small_grid, n_steps=10, dt=86400.0,
                                     heat_flux={"mean": mean_field, "amplitude": 0.0})
        arr = np.array(seq.heat_flux)
        np.testing.assert_allclose(arr[:, 0, 0], 50.0, rtol=1e-5)
        np.testing.assert_allclose(arr[:, 1, 0], 20.0, rtol=1e-5)

    def test_wrong_shape_raises(self, small_grid):
        """Array with wrong shape raises ValueError."""
        bad_array = np.ones((7, 7), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            make_synthetic_forcing(small_grid, n_steps=5, dt=300.0,
                                   heat_flux=bad_array)

    def test_negative_period_raises(self, small_grid):
        """Dict with negative period raises ValueError."""
        with pytest.raises(ValueError, match="period"):
            make_synthetic_forcing(small_grid, n_steps=5, dt=300.0,
                                   heat_flux={"period": -1.0})


# ---------------------------------------------------------------------------
# Group 2 — make_forcing_sequence
# ---------------------------------------------------------------------------

def _make_snapshot(Nx, Ny, hf_val, fw_val=0.0, tx_val=0.0, ty_val=0.0):
    """Helper: build a constant (Nx, Ny) SurfaceForcing snapshot."""
    return SurfaceForcing(
        heat_flux = jnp.full((Nx, Ny), hf_val, dtype=jnp.float32),
        fw_flux   = jnp.full((Nx, Ny), fw_val, dtype=jnp.float32),
        tau_x     = jnp.full((Nx, Ny), tx_val, dtype=jnp.float32),
        tau_y     = jnp.full((Nx, Ny), ty_val, dtype=jnp.float32),
    )


class TestMakeForcingSequence:

    def test_output_shape(self, small_grid):
        """Output shape is (n_steps, Nx, Ny) for every field."""
        Nx, Ny = small_grid.Nx, small_grid.Ny
        snapshots = [
            (0.0,      _make_snapshot(Nx, Ny, -100.0)),
            (86400.0,  _make_snapshot(Nx, Ny,    0.0)),
        ]
        seq = make_forcing_sequence(snapshots, n_steps=288, dt=300.0)
        for field in (seq.heat_flux, seq.fw_flux, seq.tau_x, seq.tau_y):
            assert field.shape == (288, Nx, Ny)

    def test_linear_at_nodes(self, small_grid):
        """Linear interp: values at snapshot times match the snapshot values."""
        Nx, Ny = small_grid.Nx, small_grid.Ny
        # Two snapshots: t=0 → -100, t=86400 → 0
        # model steps at dt=86400: step 1 → t=86400 (second snapshot)
        snapshots = [
            (0.0,     _make_snapshot(Nx, Ny, -100.0)),
            (86400.0, _make_snapshot(Nx, Ny,    0.0)),
        ]
        seq = make_forcing_sequence(snapshots, n_steps=1, dt=86400.0)
        # Step 1 corresponds to t=86400 → should be 0.0
        np.testing.assert_allclose(np.array(seq.heat_flux[0]), 0.0, atol=1e-4)

    def test_linear_midpoint(self, small_grid):
        """Linear interp: midpoint between two snapshots is their average."""
        Nx, Ny = small_grid.Nx, small_grid.Ny
        snapshots = [
            (0.0,     _make_snapshot(Nx, Ny, -100.0)),
            (2.0,     _make_snapshot(Nx, Ny,    0.0)),
        ]
        # 3 steps at dt=1: t=1,2,3 → interp at t=1 → midpoint = -50
        seq = make_forcing_sequence(snapshots, n_steps=3, dt=1.0)
        np.testing.assert_allclose(np.array(seq.heat_flux[0]), -50.0, atol=1e-4)

    def test_nearest_interpolation(self, small_grid):
        """Nearest interp: each step gets the closer snapshot's value."""
        Nx, Ny = small_grid.Nx, small_grid.Ny
        snapshots = [
            (0.0, _make_snapshot(Nx, Ny, -100.0)),
            (4.0, _make_snapshot(Nx, Ny,    50.0)),
        ]
        # Steps at dt=1: t=1,2,3,4,5,6
        # Nearest to t=1: src t=0 (dist=1) vs t=4 (dist=3) → src=0 → -100
        # Nearest to t=3: src t=0 (dist=3) vs t=4 (dist=1) → src=4 →  50
        seq = make_forcing_sequence(snapshots, n_steps=6, dt=1.0, interp="nearest")
        arr = np.array(seq.heat_flux)
        np.testing.assert_allclose(arr[0], -100.0, atol=1e-4)  # t=1, nearest to src 0
        np.testing.assert_allclose(arr[2],   50.0, atol=1e-4)  # t=3, nearest to src 4

    def test_cyclic_wraps(self, small_grid):
        """Cyclic interp: interior steps in the second cycle match the first cycle."""
        Nx, Ny = small_grid.Nx, small_grid.Ny
        T = 10.0  # period
        snapshots = [
            (0.0, _make_snapshot(Nx, Ny, -100.0)),
            (T,   _make_snapshot(Nx, Ny,    0.0)),
        ]
        # 20 steps at dt=1: tgt_times = 1,2,...,20
        # Cyclic wraps via t % 10:
        #   t=1..9  → wrapped=1..9   (first cycle interior)
        #   t=10    → wrapped=0      (boundary, snaps to t=0 value = -100)
        #   t=11..19 → wrapped=1..9  (second cycle interior, same as first)
        #   t=20    → wrapped=0      (boundary again)
        # Steps 0..8 (t=1..9) and steps 10..18 (t=11..19) should be identical.
        seq_cyclic = make_forcing_sequence(snapshots, n_steps=20, dt=1.0, interp="cyclic")
        arr = np.array(seq_cyclic.heat_flux)
        # Interior of first and second cycles must match
        np.testing.assert_allclose(arr[:9], arr[10:19], atol=1e-4)
        # Boundary step (t=10, wrapped to t=0) equals the first snapshot value
        np.testing.assert_allclose(arr[9], -100.0, atol=1e-4)

    def test_requires_two_snapshots(self, small_grid):
        """Fewer than 2 snapshots raises ValueError."""
        Nx, Ny = small_grid.Nx, small_grid.Ny
        with pytest.raises(ValueError, match="2 snapshots"):
            make_forcing_sequence(
                [(0.0, _make_snapshot(Nx, Ny, 0.0))],
                n_steps=10, dt=300.0,
            )

    def test_invalid_interp_mode(self, small_grid):
        """Unknown interp mode raises ValueError."""
        Nx, Ny = small_grid.Nx, small_grid.Ny
        snapshots = [
            (0.0, _make_snapshot(Nx, Ny, 0.0)),
            (1.0, _make_snapshot(Nx, Ny, 1.0)),
        ]
        with pytest.raises(ValueError, match="interp"):
            make_forcing_sequence(snapshots, n_steps=5, dt=1.0, interp="spline")


# ---------------------------------------------------------------------------
# Group 3 — Integration: result plugs into timeStepping.run()
# ---------------------------------------------------------------------------

class TestForcingIntegration:

    def test_synthetic_forcing_runs_without_error(self, small_grid, rest_state,
                                                   default_params):
        """make_synthetic_forcing output can be passed to run() without error."""
        n_steps = 10
        seq = make_synthetic_forcing(
            small_grid, n_steps=n_steps, dt=float(default_params.dt),
            heat_flux={"mean": -50.0, "amplitude": 30.0, "period": _YEAR},
            tau_x=-0.05,
        )
        assert seq.heat_flux.shape == (n_steps, small_grid.Nx, small_grid.Ny)
        final, _ = run(rest_state, small_grid, default_params,
                       n_steps=n_steps, forcing_sequence=seq, save_history=False)
        assert jnp.all(jnp.isfinite(final.T))
        assert jnp.all(jnp.isfinite(final.eta))

    def test_sequence_forcing_runs_without_error(self, small_grid, rest_state,
                                                  default_params):
        """make_forcing_sequence output can be passed to run() without error."""
        Nx, Ny  = small_grid.Nx, small_grid.Ny
        n_steps = 10
        dt      = float(default_params.dt)
        snapshots = [
            (0.0,        _make_snapshot(Nx, Ny, -80.0, tx_val=-0.05)),
            (n_steps*dt, _make_snapshot(Nx, Ny, -20.0, tx_val=-0.02)),
        ]
        seq = make_forcing_sequence(snapshots, n_steps=n_steps, dt=dt)
        final, _ = run(rest_state, small_grid, default_params,
                       n_steps=n_steps, forcing_sequence=seq, save_history=False)
        assert jnp.all(jnp.isfinite(final.T))

    def test_synthetic_heat_warms_ocean(self, small_grid, rest_state,
                                         default_params):
        """Positive heat flux raises surface temperature."""
        n_steps = 288  # 1 day
        seq = make_synthetic_forcing(
            small_grid, n_steps=n_steps, dt=float(default_params.dt),
            heat_flux=200.0,   # strong positive flux → warming
        )
        final, _ = run(rest_state, small_grid, default_params,
                       n_steps=n_steps, forcing_sequence=seq, save_history=False)
        T_surf_final = float(jnp.mean(final.T[:, :, 0]))
        T_surf_init  = float(jnp.mean(rest_state.T[:, :, 0]))
        assert T_surf_final > T_surf_init, \
            f"Positive heat flux should warm SST; got {T_surf_init:.4f} -> {T_surf_final:.4f}"
