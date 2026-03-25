"""
Tests for run_simulation.py (v2)
=================================
Four smoke tests covering the key v2 behaviours:

  1. Basic end-to-end run — small grid, partial chunk (n_steps not divisible
     by chunk_size), NetCDF output dimensions and NaN status.
  2. Partial-chunk correctness — final time equals n_steps * dt.
  3. save_interval / chunk_size decoupling — saves occur at the right
     number of records when the two parameters are coprime.
  4. NaN abort — run_simulation exits with code 1 when NaN is injected
     into the state after the first chunk.

Running
-------
    pytest OceanJAX/tests/test_run_simulation.py -v
"""

from __future__ import annotations

import sys
import os
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr

# run_simulation.py lives at the project root; make sure it is importable
# when pytest is invoked from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from run_simulation import main
from OceanJAX.timeStepping import run as real_ocean_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMALL_GRID = ["--nx", "4", "--ny", "4", "--nz", "3",
               "--lon_min", "0", "--lon_max", "4",
               "--lat_min", "10", "--lat_max", "14",
               "--depth_max", "300"]


def _run(tmp_path, extra_args):
    """Run main() with a temporary output file and return the open Dataset."""
    out = str(tmp_path / "out.nc")
    main(_SMALL_GRID + ["--output", out] + extra_args)
    return nc.Dataset(out, "r")


# ---------------------------------------------------------------------------
# Test 1 — basic end-to-end run
# ---------------------------------------------------------------------------

class TestBasicRun:
    """Small grid, n_steps=20, chunk_size=7 (partial chunk of 6)."""

    def test_output_file_created(self, tmp_path):
        out = str(tmp_path / "out.nc")
        main(_SMALL_GRID + ["--output", out,
                             "--n_steps", "20", "--chunk_size", "7",
                             "--save_interval", "10"])
        assert os.path.exists(out)

    def test_dimensions(self, tmp_path):
        ds = _run(tmp_path, ["--n_steps", "20", "--chunk_size", "7",
                              "--save_interval", "10"])
        with ds:
            assert ds.dimensions["x"].size == 4
            assert ds.dimensions["y"].size == 4
            assert ds.dimensions["z"].size == 3

    def test_T_shape(self, tmp_path):
        ds = _run(tmp_path, ["--n_steps", "20", "--chunk_size", "7",
                              "--save_interval", "10"])
        with ds:
            # time records: t=0 + exact saves at step 10 and step 20 = 3 total
            n_rec = ds.variables["T"].shape[0]
            assert n_rec >= 2
            assert ds.variables["T"].shape[1:] == (4, 4, 3)

    def test_no_nan(self, tmp_path):
        ds = _run(tmp_path, ["--n_steps", "20", "--chunk_size", "7",
                              "--save_interval", "10"])
        with ds:
            T = ds.variables["T"][:]
            assert not np.any(np.isnan(T))


# ---------------------------------------------------------------------------
# Test 2 — partial chunk: final time is correct
# ---------------------------------------------------------------------------

class TestPartialChunk:
    """n_steps=13, chunk_size=5 → two full chunks (10) + one partial (3)."""

    def test_final_time(self, tmp_path):
        dt = 900.0
        n_steps = 13
        # save_interval=n_steps ensures exactly one save at the final step.
        ds = _run(tmp_path, ["--n_steps", str(n_steps), "--chunk_size", "5",
                              "--save_interval", str(n_steps), "--dt", str(dt)])
        with ds:
            times = np.array(ds.variables["time"][:])
        # records: t=0 (initial) + t=n_steps*dt (final) = 2 total
        assert len(times) == 2
        assert float(times[-1]) == pytest.approx(n_steps * dt)

    def test_completes_without_error(self, tmp_path):
        """Simply verify no exception is raised."""
        _run(tmp_path, ["--n_steps", "13", "--chunk_size", "5",
                        "--save_interval", "100"])


# ---------------------------------------------------------------------------
# Test 3 — save_interval / chunk_size decoupled
# ---------------------------------------------------------------------------

class TestSaveIntervalDecoupled:
    """
    save_interval=3, chunk_size=5, n_steps=15.
    Effective chunk = min(5, 3) = 3; saves at exact steps 3, 6, 9, 12, 15.
    Expected NetCDF records: t=0 (initial) + 5 exact saves = 6 total.
    """

    def test_record_count(self, tmp_path):
        ds = _run(tmp_path, ["--n_steps", "15", "--chunk_size", "5",
                              "--save_interval", "3"])
        with ds:
            n_rec = ds.variables["time"].shape[0]
        assert n_rec == 6

    def test_time_values(self, tmp_path):
        dt = 900.0
        ds = _run(tmp_path, ["--n_steps", "15", "--chunk_size", "5",
                              "--save_interval", "3", "--dt", str(dt)])
        with ds:
            times = np.array(ds.variables["time"][:])
        expected = np.array([0, 3, 6, 9, 12, 15], dtype=np.float32) * dt
        np.testing.assert_allclose(times, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 4 — NaN abort
# ---------------------------------------------------------------------------

class TestNanAbort:
    """
    Patch ocean_run so that the second chunk returns a state with NaN in T.
    main() must call sys.exit(1).
    """

    def test_exits_with_code_1(self, tmp_path):
        out = str(tmp_path / "nan_out.nc")
        call_count = [0]

        def nan_run(state, grid, params, n_steps,
                    forcing_sequence=None, save_history=False):
            call_count[0] += 1
            final_state, history = real_ocean_run(
                state, grid, params, n_steps,
                forcing_sequence=forcing_sequence,
                save_history=save_history,
            )
            if call_count[0] >= 2:
                final_state = eqx.tree_at(
                    lambda s: s.T,
                    final_state,
                    jnp.full_like(final_state.T, float("nan")),
                )
            return final_state, history

        # jax.disable_jit() forces every run_jit call to execute the Python
        # function directly, so the mock is invoked on each chunk and NaN
        # injection takes effect on the second call.
        with jax.disable_jit():
            with patch("run_simulation.ocean_run", nan_run):
                with pytest.raises(SystemExit) as exc_info:
                    main(_SMALL_GRID + ["--output", out,
                                        "--n_steps", "20", "--chunk_size", "5"])

        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# ORAS5 initialisation smoke tests
# ---------------------------------------------------------------------------

# Synthetic ORAS5 source grid — large enough to cover the target domain below.
_ORAS5_LON   = np.array([0.0, 1.0, 2.0, 3.0, 4.0],            dtype=np.float64)
_ORAS5_LAT   = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)
_ORAS5_DEPTH = np.array([10.0, 50.0, 200.0],                   dtype=np.float64)

# Target domain within the ORAS5 source domain (no out-of-bounds warning)
_ORAS5_GRID = [
    "--nx", "3", "--ny", "4", "--nz", "2",
    "--lon_min", "1", "--lon_max", "3",
    "--lat_min", "11", "--lat_max", "14",
    "--depth_max", "150",
]


def _write_oras5_nc(path, with_uv: bool = False, with_eta: bool = False) -> None:
    """Write a minimal synthetic ORAS5-like NetCDF for smoke testing."""
    Nz, Ny, Nx = len(_ORAS5_DEPTH), len(_ORAS5_LAT), len(_ORAS5_LON)
    T = np.zeros((1, Nz, Ny, Nx), dtype=np.float32)
    for k in range(Nz):
        T[0, k] = 20.0 - k * 5.0   # 20, 15, 10 °C
    S   = np.full((1, Nz, Ny, Nx), 35.0, dtype=np.float32)
    eta = np.zeros((1, Ny, Nx), dtype=np.float32)
    uv  = np.zeros((1, Nz, Ny, Nx), dtype=np.float32)

    dims3 = ("time", "depth", "lat", "lon")
    dims2 = ("time", "lat", "lon")
    data_vars = {
        "thetao": (dims3, T),
        "so":     (dims3, S),
    }
    if with_uv:
        data_vars["uo"] = (dims3, uv)
        data_vars["vo"] = (dims3, uv)
    if with_eta:
        data_vars["zos"] = (dims2, eta)

    ds = xr.Dataset(data_vars, coords={
        "lon":   ("lon",   _ORAS5_LON),
        "lat":   ("lat",   _ORAS5_LAT),
        "depth": ("depth", _ORAS5_DEPTH),
        "time":  ("time",  np.array([0.0])),
    })
    ds.to_netcdf(path)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

class TestValidation:
    """_validate_args must reject bad grid / run-control arguments."""

    def _bad(self, tmp_path, extra_args):
        """Assert that main() exits with code 1 for the given bad args."""
        out = str(tmp_path / "out.nc")
        with pytest.raises(SystemExit) as exc_info:
            main(_SMALL_GRID + ["--output", out, "--n_steps", "5"] + extra_args)
        assert exc_info.value.code == 1

    def test_dt_zero(self, tmp_path):
        self._bad(tmp_path, ["--dt", "0"])

    def test_dt_negative(self, tmp_path):
        self._bad(tmp_path, ["--dt", "-1"])

    def test_depth_max_zero(self, tmp_path):
        self._bad(tmp_path, ["--depth_max", "0"])

    def test_lon_inverted(self, tmp_path):
        self._bad(tmp_path, ["--lon_min", "10", "--lon_max", "5"])

    def test_lat_inverted(self, tmp_path):
        self._bad(tmp_path, ["--lat_min", "20", "--lat_max", "10"])

    def test_lon_equal(self, tmp_path):
        self._bad(tmp_path, ["--lon_min", "5", "--lon_max", "5"])


class TestOras5Init:
    """Smoke tests for --init oras5 in run_simulation.py."""

    def test_ts_only_no_nan(self, tmp_path):
        """ORAS5 init with T/S only (u/v/eta zero-filled) must produce no NaN."""
        nc_path = str(tmp_path / "oras5.nc")
        out     = str(tmp_path / "out.nc")
        _write_oras5_nc(nc_path)
        main(_ORAS5_GRID + [
            "--output", out,
            "--init", "oras5", "--oras5_path", nc_path,
            "--n_steps", "5", "--chunk_size", "5", "--save_interval", "5",
        ])
        with nc.Dataset(out, "r") as ds:
            T = ds.variables["T"][:]
            S = ds.variables["S"][:]
            assert not np.any(np.isnan(T)), "NaN in T after ORAS5 init"
            assert not np.any(np.isnan(S)), "NaN in S after ORAS5 init"

    def test_ts_ranges_plausible(self, tmp_path):
        """T must stay within synthetic source range after a few steps."""
        nc_path = str(tmp_path / "oras5.nc")
        out     = str(tmp_path / "out.nc")
        _write_oras5_nc(nc_path)
        main(_ORAS5_GRID + [
            "--output", out,
            "--init", "oras5", "--oras5_path", nc_path,
            "--n_steps", "5", "--chunk_size", "5", "--save_interval", "5",
        ])
        with nc.Dataset(out, "r") as ds:
            T = ds.variables["T"][:]
            S = ds.variables["S"][:]
        # Source T ∈ [10, 20]; allow modest drift over 5 steps
        assert float(T.min()) > 0.0,  "T fell below 0 — likely fill contamination"
        assert float(T.max()) < 30.0, "T exceeded 30 — likely fill contamination"
        assert float(S.min()) > 30.0, "S fell below 30"
        assert float(S.max()) < 40.0, "S exceeded 40"

    def test_full_uveta_no_nan(self, tmp_path):
        """ORAS5 init with all five variables must produce no NaN."""
        nc_path = str(tmp_path / "oras5_full.nc")
        out     = str(tmp_path / "out.nc")
        _write_oras5_nc(nc_path, with_uv=True, with_eta=True)
        main(_ORAS5_GRID + [
            "--output", out,
            "--init", "oras5", "--oras5_path", nc_path,
            "--n_steps", "5", "--chunk_size", "5", "--save_interval", "5",
        ])
        with nc.Dataset(out, "r") as ds:
            T   = ds.variables["T"][:]
            eta = ds.variables["eta"][:]
        assert not np.any(np.isnan(T)),   "NaN in T"
        assert not np.any(np.isnan(eta)), "NaN in eta"

    def test_missing_oras5_path_exits(self, tmp_path):
        """--init oras5 without --oras5_path must exit with code 1."""
        out = str(tmp_path / "out.nc")
        with pytest.raises(SystemExit) as exc_info:
            main(_ORAS5_GRID + [
                "--output", out,
                "--init", "oras5",
                "--n_steps", "5",
            ])
        assert exc_info.value.code == 1

    def test_init_mode_in_metadata(self, tmp_path):
        """Output NetCDF must record init_mode = 'oras5'."""
        nc_path = str(tmp_path / "oras5.nc")
        out     = str(tmp_path / "out.nc")
        _write_oras5_nc(nc_path)
        main(_ORAS5_GRID + [
            "--output", out,
            "--init", "oras5", "--oras5_path", nc_path,
            "--n_steps", "5", "--chunk_size", "5", "--save_interval", "5",
        ])
        with nc.Dataset(out, "r") as ds:
            assert ds.init_mode == "oras5"
