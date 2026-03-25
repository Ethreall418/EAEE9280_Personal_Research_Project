"""
Tests for OceanJAX.data.oras5
==============================
All tests use synthetic in-memory NetCDF files; no real ORAS5 download required.

Coverage:
  TestReadOras5      — file I/O, alias resolution, coordinate normalisation,
                       optional-variable contract, error paths.
  TestRegridToModel  — interpolation shapes, NaN-fill strategy,
                       None u/v/eta → zero fields, resolution warning,
                       OceanGrid mask application.

Running
-------
    pytest OceanJAX/tests/test_oras5.py -v
"""

from __future__ import annotations

import warnings
import numpy as np
import pytest
import xarray as xr

from OceanJAX.grid import OceanGrid
from OceanJAX.data.oras5 import read_oras5, regrid_to_model, load_oras5


# ---------------------------------------------------------------------------
# Shared synthetic-data helpers
# ---------------------------------------------------------------------------

# Source grid dimensions
_SRC_LON   = np.array([0.0, 1.0, 2.0, 3.0, 4.0],         dtype=np.float64)
_SRC_LAT   = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)
_SRC_DEPTH = np.array([10.0, 50.0, 200.0],                  dtype=np.float64)

_NX_SRC = len(_SRC_LON)
_NY_SRC = len(_SRC_LAT)
_NZ_SRC = len(_SRC_DEPTH)


def _make_T() -> np.ndarray:
    """Synthetic temperature: linearly varies with depth (Nz, Ny, Nx)."""
    T = np.zeros((_NZ_SRC, _NY_SRC, _NX_SRC), dtype=np.float32)
    for k in range(_NZ_SRC):
        T[k] = 20.0 - k * 5.0   # 20, 15, 10 °C
    return T


def _make_S() -> np.ndarray:
    S = np.full((_NZ_SRC, _NY_SRC, _NX_SRC), 35.0, dtype=np.float32)
    return S


def _write_nc(
    path,
    T_name:   str = "thetao",
    S_name:   str = "so",
    u_name:   str | None = None,
    v_name:   str | None = None,
    eta_name: str | None = None,
    lat:      np.ndarray | None = None,
    add_time: bool = True,
    nan_mask: np.ndarray | None = None,  # (Nz, Ny, Nx) bool — set these T cells to NaN
) -> None:
    """Write a minimal synthetic ORAS5-like NetCDF to *path*."""
    if lat is None:
        lat = _SRC_LAT

    T = _make_T()
    S = _make_S()

    if nan_mask is not None:
        T = T.copy()
        T[nan_mask] = np.nan

    if add_time:
        dims_3d  = ("time", "depth", "lat", "lon")
        dims_2d  = ("time", "lat",   "lon")
        T_data   = T[np.newaxis]
        S_data   = S[np.newaxis]
        u_data   = np.zeros_like(T)[np.newaxis]
        v_data   = np.zeros_like(T)[np.newaxis]
        eta_data = np.zeros((_NY_SRC, _NX_SRC), dtype=np.float32)[np.newaxis]
    else:
        dims_3d  = ("depth", "lat", "lon")
        dims_2d  = ("lat",   "lon")
        T_data   = T
        S_data   = S
        u_data   = np.zeros_like(T)
        v_data   = np.zeros_like(T)
        eta_data = np.zeros((_NY_SRC, _NX_SRC), dtype=np.float32)

    coords = {
        "lon":   ("lon",   _SRC_LON),
        "lat":   ("lat",   lat),
        "depth": ("depth", _SRC_DEPTH),
    }
    if add_time:
        coords["time"] = ("time", np.array([0.0]))

    data_vars = {
        T_name: (dims_3d, T_data),
        S_name: (dims_3d, S_data),
    }
    if u_name is not None:
        data_vars[u_name] = (dims_3d, u_data)
    if v_name is not None:
        data_vars[v_name] = (dims_3d, v_data)
    if eta_name is not None:
        data_vars[eta_name] = (dims_2d, eta_data)

    ds = xr.Dataset(data_vars, coords=coords)
    ds.to_netcdf(path)


def _target_grid() -> OceanGrid:
    """Small OceanJAX grid entirely within the source domain."""
    depth_levels = np.array([15.0, 100.0], dtype=np.float64)
    return OceanGrid.create(
        lon_bounds=(1.0, 3.0),
        lat_bounds=(11.0, 14.0),
        depth_levels=depth_levels,
        Nx=3,
        Ny=4,
    )


# ---------------------------------------------------------------------------
# TestReadOras5
# ---------------------------------------------------------------------------

class TestReadOras5:

    def test_required_only_uveta_are_none(self, tmp_path):
        """File with only T and S: u, v, eta must be None (not absent keys)."""
        nc = tmp_path / "req_only.nc"
        _write_nc(nc)
        raw = read_oras5(nc)

        assert raw["u"]   is None
        assert raw["v"]   is None
        assert raw["eta"] is None

    def test_all_variables_non_none(self, tmp_path):
        """File with all 5 variables: none should be None."""
        nc = tmp_path / "all_vars.nc"
        _write_nc(nc, u_name="uo", v_name="vo", eta_name="zos")
        raw = read_oras5(nc)

        assert raw["u"]   is not None
        assert raw["v"]   is not None
        assert raw["eta"] is not None

    def test_alias_thetao_oras(self, tmp_path):
        """thetao_oras / so_oras aliases are resolved correctly."""
        nc = tmp_path / "alias_oras.nc"
        _write_nc(nc, T_name="thetao_oras", S_name="so_oras")
        raw = read_oras5(nc)
        assert raw["T"] is not None
        assert raw["T"].shape == (_NZ_SRC, _NY_SRC, _NX_SRC)

    def test_alias_votemper(self, tmp_path):
        """votemper / vosaline (NEMO-style) aliases are resolved."""
        nc = tmp_path / "alias_nemo.nc"
        _write_nc(nc, T_name="votemper", S_name="vosaline")
        raw = read_oras5(nc)
        assert raw["T"] is not None

    def test_negative_depth_flipped(self, tmp_path):
        """Negative depth coordinates are flipped to positive-downward."""
        # Write a file then patch the depth coordinate to negative
        nc = tmp_path / "neg_depth.nc"
        _write_nc(nc)
        ds = xr.open_dataset(nc).load()
        ds = ds.assign_coords(depth=-ds["depth"])
        ds.to_netcdf(tmp_path / "neg_depth2.nc")

        raw = read_oras5(tmp_path / "neg_depth2.nc")
        assert np.all(raw["depth"] > 0), "depth should be positive downward"

    def test_descending_lat_sorted(self, tmp_path):
        """Descending latitude coordinates are sorted to ascending."""
        nc = tmp_path / "desc_lat.nc"
        _write_nc(nc, lat=_SRC_LAT[::-1])   # descending
        raw = read_oras5(nc)

        assert np.all(np.diff(raw["lat"]) > 0), "lat should be ascending"
        assert raw["T"].shape == (_NZ_SRC, _NY_SRC, _NX_SRC)

    def test_descending_lat_values_consistent(self, tmp_path):
        """After sorting, T values must match those from an ascending-lat file."""
        nc_asc  = tmp_path / "asc.nc"
        nc_desc = tmp_path / "desc.nc"
        _write_nc(nc_asc,  lat=_SRC_LAT)
        _write_nc(nc_desc, lat=_SRC_LAT[::-1])
        raw_asc  = read_oras5(nc_asc)
        raw_desc = read_oras5(nc_desc)

        np.testing.assert_allclose(raw_asc["T"], raw_desc["T"], atol=1e-5)

    def test_2d_coords_raises(self, tmp_path):
        """2-D coordinate arrays (curvilinear grid) must raise ValueError."""
        nc = tmp_path / "curv.nc"
        # Write a file with 2-D lon/lat coords
        lon2d = np.tile(_SRC_LON, (_NY_SRC, 1))
        lat2d = np.tile(_SRC_LAT[:, np.newaxis], (1, _NX_SRC))
        T_data = _make_T()[np.newaxis]

        ds = xr.Dataset(
            {"thetao": (("time", "depth", "y", "x"), T_data),
             "so":     (("time", "depth", "y", "x"), T_data)},
            coords={
                "lon":   (("y", "x"), lon2d),
                "lat":   (("y", "x"), lat2d),
                "depth": ("depth",   _SRC_DEPTH),
                "time":  ("time",    [0.0]),
            },
        )
        ds.to_netcdf(nc)
        with pytest.raises(ValueError, match="1-D"):
            read_oras5(nc)

    def test_missing_required_raises(self, tmp_path):
        """File with no T-like variable must raise KeyError."""
        nc = tmp_path / "no_T.nc"
        ds = xr.Dataset(
            {"so": (("time", "depth", "lat", "lon"), _make_S()[np.newaxis])},
            coords={
                "lon": ("lon", _SRC_LON),
                "lat": ("lat", _SRC_LAT),
                "depth": ("depth", _SRC_DEPTH),
                "time":  ("time",  [0.0]),
            },
        )
        ds.to_netcdf(nc)
        with pytest.raises(KeyError):
            read_oras5(nc)

    def test_return_keys_always_present(self, tmp_path):
        """All eight dict keys are present regardless of which vars exist."""
        nc = tmp_path / "keys.nc"
        _write_nc(nc)
        raw = read_oras5(nc)
        for key in ("T", "S", "u", "v", "eta", "lon", "lat", "depth"):
            assert key in raw, f"Key '{key}' missing from read_oras5 output"

    def test_coordinate_shapes(self, tmp_path):
        """lon/lat/depth are 1-D with the expected lengths."""
        nc = tmp_path / "coords.nc"
        _write_nc(nc)
        raw = read_oras5(nc)
        assert raw["lon"].shape   == (_NX_SRC,)
        assert raw["lat"].shape   == (_NY_SRC,)
        assert raw["depth"].shape == (_NZ_SRC,)


# ---------------------------------------------------------------------------
# TestRegridToModel
# ---------------------------------------------------------------------------

class TestRegridToModel:

    def test_output_shapes(self, tmp_path):
        """T, S, u, v have shape (Nx, Ny, Nz); eta has shape (Nx, Ny)."""
        nc = tmp_path / "shapes.nc"
        _write_nc(nc, u_name="uo", v_name="vo", eta_name="zos")
        grid = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)

        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        assert state.T.shape   == (Nx, Ny, Nz)
        assert state.S.shape   == (Nx, Ny, Nz)
        assert state.u.shape   == (Nx, Ny, Nz)
        assert state.v.shape   == (Nx, Ny, Nz)
        assert state.eta.shape == (Nx, Ny)

    def test_no_nan_in_output(self, tmp_path):
        """Regridded state must contain no NaN."""
        nc = tmp_path / "nonan.nc"
        _write_nc(nc)
        grid = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)

        assert not np.any(np.isnan(np.array(state.T))), "NaN in T"
        assert not np.any(np.isnan(np.array(state.S))), "NaN in S"

    def test_nan_fill_linear_then_nn(self, tmp_path):
        """NaN cells in the source (coast mask) must be filled by the three-tier strategy."""
        # Set all source T values at the top two cells of j=0 to NaN
        nan_mask = np.zeros((_NZ_SRC, _NY_SRC, _NX_SRC), dtype=bool)
        nan_mask[:, 0, :] = True   # entire southern row is NaN

        nc = tmp_path / "nan_src.nc"
        _write_nc(nc, nan_mask=nan_mask)
        grid = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid, T_fill=5.0)

        T = np.array(state.T)
        assert not np.any(np.isnan(T)), "NaN survived three-tier fill strategy"
        # All ocean cells should have T > 0 (fill=5°, real data ≥ 10°)
        ocean = np.array(grid.mask_c, dtype=bool)
        assert np.all(T[ocean] > 0.0)

    def test_none_uveta_gives_zero_fields(self, tmp_path):
        """raw u/v/eta = None must produce zero u, v, eta in the OceanState."""
        nc = tmp_path / "ts_only.nc"
        _write_nc(nc)   # no u/v/eta
        grid = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)

        assert np.allclose(np.array(state.u),   0.0), "u should be zero"
        assert np.allclose(np.array(state.v),   0.0), "v should be zero"
        assert np.allclose(np.array(state.eta), 0.0), "eta should be zero"

    def test_resolution_warning(self, tmp_path):
        """Finer target than source grid must trigger UserWarning."""
        nc = tmp_path / "fine.nc"
        _write_nc(nc)
        raw = read_oras5(nc)

        # Target grid at ~0.1° — finer than the 1° synthetic source
        depth_levels = np.array([15.0, 100.0], dtype=np.float64)
        fine_grid = OceanGrid.create(
            lon_bounds=(1.0, 2.5),
            lat_bounds=(11.0, 13.5),
            depth_levels=depth_levels,
            Nx=15,
            Ny=25,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            regrid_to_model(raw, fine_grid)

        categories = [str(w.category) for w in caught]
        assert any("UserWarning" in c for c in categories), (
            "Expected UserWarning for upscaling; got: " + str(categories)
        )

    def test_masks_applied(self, tmp_path):
        """Land points (mask_c == 0) must be zero in regridded T."""
        nc = tmp_path / "mask.nc"
        _write_nc(nc)

        depth_levels = np.array([15.0, 100.0, 300.0], dtype=np.float64)
        bathy = np.full((3, 4), 200.0)   # most cells: 200 m → only k=0,1 are wet
        grid = OceanGrid.create(
            lon_bounds=(1.0, 3.0),
            lat_bounds=(11.0, 14.0),
            depth_levels=depth_levels,
            Nx=3,
            Ny=4,
            bathymetry=bathy,
        )
        state = regrid_to_model(read_oras5(nc), grid)

        T = np.array(state.T)
        mask = np.array(grid.mask_c)
        # T must be zero wherever mask_c == 0
        assert np.allclose(T[mask == 0], 0.0), (
            "Non-zero T values in land cells after regridding"
        )

    def test_no_coastal_contamination(self, tmp_path):
        """
        NaN-aware fill must not bleed fill_value into coast-adjacent ocean cells.

        Setup: all source T values are 10.0 except one coastal column (i=0)
        which is entirely NaN (land).  A large fill_value=999 is passed.
        The target grid sits well inside the ocean region (lon >= 1.0) so
        none of its cells should receive a value blended with 999.
        With the old fill-then-interpolate approach, coast-adjacent target
        points would come out ≈ (10*w + 999*(1-w)) >> 10.  With normalised
        convolution the answer should be ≈ 10.
        """
        nan_mask = np.zeros((_NZ_SRC, _NY_SRC, _NX_SRC), dtype=bool)
        nan_mask[:, :, 0] = True    # entire western column is NaN / land

        nc = tmp_path / "coastal.nc"
        _write_nc(nc, nan_mask=nan_mask)
        grid = _target_grid()   # lon in [1, 3], safely away from lon=0 coast

        state = regrid_to_model(read_oras5(nc), grid, T_fill=999.0)
        T = np.array(state.T)
        ocean = np.array(grid.mask_c, dtype=bool)

        # All ocean cells should be close to real ocean values (10–20 °C),
        # NOT contaminated by the fill_value of 999.
        assert np.all(T[ocean] < 50.0), (
            f"Coastal contamination detected: max T = {T[ocean].max():.1f} "
            "(expected ≈ 10–20, fill=999)"
        )

    def test_uv_zero_below_source_depth(self, tmp_path):
        """u and v must be exactly zero at target levels below src_depth.max()."""
        nc = tmp_path / "deep_uv.nc"
        _write_nc(nc, u_name="uo", v_name="vo", eta_name="zos")

        # Target grid with a level (300 m) deeper than _SRC_DEPTH.max() = 200 m
        depth_levels = np.array([15.0, 300.0], dtype=np.float64)
        grid = OceanGrid.create(
            lon_bounds=(1.0, 3.0),
            lat_bounds=(11.0, 14.0),
            depth_levels=depth_levels,
            Nx=3,
            Ny=4,
        )
        state = regrid_to_model(read_oras5(nc), grid)
        u = np.array(state.u)
        v = np.array(state.v)

        # k=1 corresponds to 300 m, which is below the source max of 200 m
        assert np.allclose(u[:, :, 1], 0.0), "u at 300 m (below src) should be zero"
        assert np.allclose(v[:, :, 1], 0.0), "v at 300 m (below src) should be zero"

    def test_load_oras5_wrapper(self, tmp_path):
        """load_oras5 convenience wrapper produces same result as two-layer call."""
        nc = tmp_path / "wrap.nc"
        _write_nc(nc)
        grid = _target_grid()

        state_direct = regrid_to_model(read_oras5(nc), grid)
        state_wrap   = load_oras5(nc, grid)

        np.testing.assert_allclose(np.array(state_wrap.T), np.array(state_direct.T))
        np.testing.assert_allclose(np.array(state_wrap.S), np.array(state_direct.S))
