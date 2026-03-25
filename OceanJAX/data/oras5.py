"""
OceanJAX ORAS5 data loader
===========================
Loads the Copernicus Marine Service ORAS5 ocean reanalysis product
(GLOBAL_MULTIYEAR_PHY_001_030) and regrids it onto an OceanJAX model grid.

Targeted format
---------------
Regular 0.25° lat-lon grid, 1-D coordinate variables, 75 vertical levels,
NetCDF-CF convention.  Native NEMO/ORCA curvilinear grids (2-D coordinate
arrays) are not supported by this loader.

Variable-name aliasing
----------------------
Copernicus variable names vary slightly across ORAS5 product versions.
The reader resolves names through a priority-ordered alias table so that
``thetao`` / ``thetao_oras`` / ``votemper`` are all accepted transparently.

Two-layer design
----------------
``read_oras5()``      — pure I/O: opens the file, extracts one time slice,
                        returns a plain dict of numpy arrays.  No JAX
                        dependency; easy to test with synthetic NetCDF files.
``regrid_to_model()`` — interpolation and masking: takes that dict and an
                        OceanGrid, returns a masked OceanState.
``load_oras5()``      — convenience wrapper that calls both layers.

Target resolution
-----------------
ORAS5 source resolution is ~0.25°.  Regridding onto a *finer* target grid
invents spurious fine-scale structure; a ``UserWarning`` is issued but
execution is not blocked.

Missing-value strategy (three-tier, NaN-aware)
-----------------------------------------------
1. Normalised convolution — data and validity mask are interpolated
   separately (data_sum / mask_sum), so coast-adjacent ocean points
   are never contaminated by fill values.
2. ``NearestNDInterpolator`` on valid source points only — fills any
   target point whose entire linear stencil is NaN (e.g. isolated
   deep cells below the bathymetry).
3. Constant ``T_fill`` / ``S_fill`` / ``0`` as unconditional fallback.

Vertical extrapolation
-----------------------
  T, S — nearest valid source level (handled automatically by tier 2).
  u, v — explicitly zeroed below the deepest ORAS5 level (extrapolating
          bottom velocities downward is physically wrong).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

from OceanJAX.grid import OceanGrid
from OceanJAX.state import OceanState, create_from_arrays


# ---------------------------------------------------------------------------
# Alias tables
# ---------------------------------------------------------------------------

_VAR_ALIASES: dict[str, list[str]] = {
    "T":   ["thetao", "thetao_oras", "votemper", "temperature"],
    "S":   ["so",     "so_oras",     "vosaline",  "salinity"],
    "u":   ["uo",     "uo_oras",     "vozocrtx",  "u"],
    "v":   ["vo",     "vo_oras",     "vomecrty",  "v"],
    "eta": ["zos",    "zos_oras",    "sossheig",  "ssh"],
}

_COORD_ALIASES: dict[str, list[str]] = {
    "lon":   ["longitude", "lon", "x", "nav_lon"],
    "lat":   ["latitude",  "lat", "y", "nav_lat"],
    "depth": ["depth", "deptht", "level", "lev", "z"],
}


def _find_var(ds: xr.Dataset, key: str, optional: bool = False) -> Optional[str]:
    """Return the first matching variable name, or None if optional and absent."""
    for name in _VAR_ALIASES[key]:
        if name in ds:
            return name
    if optional:
        return None
    raise KeyError(
        f"Cannot find variable '{key}' in dataset.  "
        f"Tried: {_VAR_ALIASES[key]}.  "
        f"Available data variables: {sorted(ds.data_vars)}"
    )


def _find_coord(ds: xr.Dataset, key: str) -> str:
    """Return the first matching coordinate name (checks both coords and data_vars)."""
    for name in _COORD_ALIASES[key]:
        if name in ds.coords or name in ds:
            return name
    raise KeyError(
        f"Cannot find coordinate '{key}' in dataset.  "
        f"Tried: {_COORD_ALIASES[key]}.  "
        f"Available coords: {sorted(ds.coords)}"
    )


def _ensure_ascending(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_arr, index_order) such that sorted_arr is strictly ascending."""
    order = np.argsort(arr)
    return arr[order], order


# ---------------------------------------------------------------------------
# Public: read_oras5
# ---------------------------------------------------------------------------

def read_oras5(
    path: str | Path,
    time_index: int = 0,
) -> dict[str, Optional[np.ndarray]]:
    """
    Read one time slice from a Copernicus Marine ORAS5 NetCDF file.

    Targets the regular 0.25° lat-lon product with 1-D coordinate variables.
    Passing a file with 2-D (curvilinear) coordinates raises ``ValueError``.

    Parameters
    ----------
    path : str or Path
        NetCDF file containing at least temperature and salinity.
    time_index : int
        Index along the time dimension to read (default 0).

    Returns
    -------
    dict with the following fixed keys.  Optional fields are **always**
    ``None`` when absent — never a missing key or an empty array:

    ``"T"``     : ``(Nz_src, Ny_src, Nx_src)`` float32 — potential temperature [°C]
    ``"S"``     : ``(Nz_src, Ny_src, Nx_src)`` float32 — practical salinity [psu]
    ``"u"``     : ``(Nz_src, Ny_src, Nx_src)`` float32, or ``None``
    ``"v"``     : ``(Nz_src, Ny_src, Nx_src)`` float32, or ``None``
    ``"eta"``   : ``(Ny_src, Nx_src)``          float32, or ``None``
    ``"lon"``   : ``(Nx_src,)`` float64 — degrees east, strictly ascending
    ``"lat"``   : ``(Ny_src,)`` float64 — degrees north, strictly ascending
    ``"depth"`` : ``(Nz_src,)`` float64 — metres positive downward, ascending

    Land / fill-value cells are returned as ``np.nan`` (xarray's
    ``mask_and_scale=True`` handles ``_FillValue`` / ``missing_value``
    automatically).
    """
    ds = xr.open_dataset(path, mask_and_scale=True)
    try:
        # --- coordinates ---
        lon_name   = _find_coord(ds, "lon")
        lat_name   = _find_coord(ds, "lat")
        depth_name = _find_coord(ds, "depth")

        lon   = np.asarray(ds[lon_name].values,   dtype=np.float64)
        lat   = np.asarray(ds[lat_name].values,   dtype=np.float64)
        depth = np.asarray(ds[depth_name].values, dtype=np.float64)

        if lon.ndim != 1 or lat.ndim != 1 or depth.ndim != 1:
            raise ValueError(
                "read_oras5 requires 1-D lon/lat/depth coordinates "
                "(Copernicus Marine regular-grid product).  "
                "Found 2-D coordinate arrays — native NEMO/ORCA curvilinear "
                "grids are not supported by this loader."
            )

        # Ensure depth is positive downward
        if np.any(depth < 0):
            depth = -depth

        # Sort all axes to strictly ascending order
        lon,   lon_idx = _ensure_ascending(lon)
        lat,   lat_idx = _ensure_ascending(lat)
        depth, dep_idx = _ensure_ascending(depth)

        # --- load and reorder a 3-D variable ---
        def _load_3d(var_name: str) -> np.ndarray:
            da = ds[var_name]
            if "time" in da.dims:
                da = da.isel(time=time_index)
            arr = np.asarray(da.values, dtype=np.float32)
            if arr.ndim != 3:
                raise ValueError(
                    f"Variable '{var_name}' has {arr.ndim} dimensions after "
                    f"time selection; expected 3 (depth, lat, lon)."
                )
            return arr[np.ix_(dep_idx, lat_idx, lon_idx)]

        # --- load and reorder a 2-D variable ---
        def _load_2d(var_name: str) -> np.ndarray:
            da = ds[var_name]
            if "time" in da.dims:
                da = da.isel(time=time_index)
            arr = np.asarray(da.values, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(
                    f"Variable '{var_name}' has {arr.ndim} dimensions after "
                    f"time selection; expected 2 (lat, lon)."
                )
            return arr[np.ix_(lat_idx, lon_idx)]

        T = _load_3d(_find_var(ds, "T"))
        S = _load_3d(_find_var(ds, "S"))

        u_name   = _find_var(ds, "u",   optional=True)
        v_name   = _find_var(ds, "v",   optional=True)
        eta_name = _find_var(ds, "eta", optional=True)

        u   = _load_3d(u_name)   if u_name   is not None else None
        v   = _load_3d(v_name)   if v_name   is not None else None
        eta = _load_2d(eta_name) if eta_name is not None else None

    finally:
        ds.close()

    return {
        "T": T, "S": S, "u": u, "v": v, "eta": eta,
        "lon": lon, "lat": lat, "depth": depth,
    }


# ---------------------------------------------------------------------------
# Private: interpolation helpers
# ---------------------------------------------------------------------------

def _normalized_conv_3d(
    src:       np.ndarray,   # (Nz, Ny, Nx), may contain NaN
    src_depth: np.ndarray,
    src_lat:   np.ndarray,
    src_lon:   np.ndarray,
    tgt_depth: np.ndarray,
    tgt_lat:   np.ndarray,
    tgt_lon:   np.ndarray,
) -> np.ndarray:
    """
    NaN-aware linear interpolation via normalised convolution.

    Instead of replacing NaN with fill_value (which contaminates coast-adjacent
    ocean points), we interpolate the data and the validity mask separately:

        data_sum  = RGI_linear(NaN → 0)
        mask_sum  = RGI_linear(valid=1, NaN=0)
        out       = data_sum / mask_sum   where mask_sum > eps

    Points where mask_sum ≤ eps receive NaN and are handled by the caller.
    """
    valid      = (~np.isnan(src)).astype(np.float64)
    src_filled = np.where(np.isnan(src), 0.0, src).astype(np.float64)

    dz, dy, dx = np.meshgrid(tgt_depth, tgt_lat, tgt_lon, indexing="ij")
    pts = np.stack([dz.ravel(), dy.ravel(), dx.ravel()], axis=-1)
    out_shape  = (len(tgt_depth), len(tgt_lat), len(tgt_lon))

    rgi_data = RegularGridInterpolator(
        (src_depth, src_lat, src_lon), src_filled,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    rgi_mask = RegularGridInterpolator(
        (src_depth, src_lat, src_lon), valid,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    data_sum = rgi_data(pts).reshape(out_shape)
    mask_sum = rgi_mask(pts).reshape(out_shape)

    eps = 1e-6
    out = np.where(mask_sum > eps, data_sum / np.maximum(mask_sum, eps), np.nan)
    return out


def _normalized_conv_2d(
    src:     np.ndarray,   # (Ny, Nx), may contain NaN
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
) -> np.ndarray:
    """2-D version of ``_normalized_conv_3d``."""
    valid      = (~np.isnan(src)).astype(np.float64)
    src_filled = np.where(np.isnan(src), 0.0, src).astype(np.float64)

    dy, dx = np.meshgrid(tgt_lat, tgt_lon, indexing="ij")
    pts = np.stack([dy.ravel(), dx.ravel()], axis=-1)
    out_shape = (len(tgt_lat), len(tgt_lon))

    rgi_data = RegularGridInterpolator(
        (src_lat, src_lon), src_filled,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    rgi_mask = RegularGridInterpolator(
        (src_lat, src_lon), valid,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    data_sum = rgi_data(pts).reshape(out_shape)
    mask_sum = rgi_mask(pts).reshape(out_shape)

    eps = 1e-6
    out = np.where(mask_sum > eps, data_sum / np.maximum(mask_sum, eps), np.nan)
    return out


def _interp_3d(
    src:        np.ndarray,
    src_lon:    np.ndarray,
    src_lat:    np.ndarray,
    src_depth:  np.ndarray,
    tgt_lon:    np.ndarray,
    tgt_lat:    np.ndarray,
    tgt_depth:  np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """
    Interpolate ``(Nz_src, Ny_src, Nx_src)`` → ``(Nz_tgt, Ny_tgt, Nx_tgt)``.

    Three-tier NaN strategy:
    1. Normalised convolution (NaN-aware linear) — no coastal contamination.
    2. Per-level ``NearestNDInterpolator`` on valid source points only —
       fills points where the local linear stencil is entirely NaN.
    3. Constant ``fill_value`` as final fallback.
    """
    out = _normalized_conv_3d(
        src, src_depth, src_lat, src_lon,
        tgt_depth, tgt_lat, tgt_lon,
    )

    needs_nn = np.isnan(out)
    if np.any(needs_nn):
        # Per-level nearest-ocean-neighbour on valid source points only
        dz_src, dy_src, dx_src = np.meshgrid(
            src_depth, src_lat, src_lon, indexing="ij"
        )
        valid_mask = ~np.isnan(src)

        dz, dy, dx = np.meshgrid(tgt_depth, tgt_lat, tgt_lon, indexing="ij")
        nn_fill = np.empty_like(out)
        for k in range(len(tgt_depth)):
            # Collect valid source points across all depths for this output level
            pts_valid = np.stack([
                dz_src[valid_mask].ravel(),
                dy_src[valid_mask].ravel(),
                dx_src[valid_mask].ravel(),
            ], axis=-1)
            vals_valid = src[valid_mask].ravel()
            if len(vals_valid) == 0:
                nn_fill[k] = fill_value
            else:
                interp_nn = NearestNDInterpolator(pts_valid, vals_valid)
                tgt_pts_k = np.stack([
                    dz[k].ravel(), dy[k].ravel(), dx[k].ravel()
                ], axis=-1)
                nn_fill[k] = interp_nn(tgt_pts_k).reshape(
                    len(tgt_lat), len(tgt_lon)
                )
        out = np.where(needs_nn, nn_fill, out)

    still_nan = np.isnan(out)
    if np.any(still_nan):
        out = np.where(still_nan, fill_value, out)

    return out.astype(np.float32)


def _interp_2d(
    src:        np.ndarray,
    src_lon:    np.ndarray,
    src_lat:    np.ndarray,
    tgt_lon:    np.ndarray,
    tgt_lat:    np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """
    Interpolate ``(Ny_src, Nx_src)`` → ``(Ny_tgt, Nx_tgt)``.
    Same three-tier NaN strategy as ``_interp_3d``.
    """
    out = _normalized_conv_2d(src, src_lat, src_lon, tgt_lat, tgt_lon)

    needs_nn = np.isnan(out)
    if np.any(needs_nn):
        dy_src, dx_src = np.meshgrid(src_lat, src_lon, indexing="ij")
        valid_mask = ~np.isnan(src)
        pts_valid  = np.stack([dy_src[valid_mask].ravel(),
                               dx_src[valid_mask].ravel()], axis=-1)
        vals_valid = src[valid_mask].ravel()

        if len(vals_valid) == 0:
            out = np.where(needs_nn, fill_value, out)
        else:
            interp_nn  = NearestNDInterpolator(pts_valid, vals_valid)
            dy, dx = np.meshgrid(tgt_lat, tgt_lon, indexing="ij")
            tgt_pts = np.stack([dy.ravel(), dx.ravel()], axis=-1)
            nn_fill = interp_nn(tgt_pts).reshape(len(tgt_lat), len(tgt_lon))
            out = np.where(needs_nn, nn_fill, out)

    still_nan = np.isnan(out)
    if np.any(still_nan):
        out = np.where(still_nan, fill_value, out)

    return out.astype(np.float32)


def _check_resolution(
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    tgt_lon: np.ndarray,
    tgt_lat: np.ndarray,
) -> None:
    src_dx = float(np.median(np.diff(src_lon)))
    src_dy = float(np.median(np.diff(src_lat)))
    tgt_dx = float(np.median(np.diff(tgt_lon))) if len(tgt_lon) > 1 else src_dx
    tgt_dy = float(np.median(np.diff(tgt_lat))) if len(tgt_lat) > 1 else src_dy
    if tgt_dx < src_dx * 0.9 or tgt_dy < src_dy * 0.9:
        warnings.warn(
            f"Target grid ({tgt_dx:.3f}° × {tgt_dy:.3f}°) is finer than the "
            f"ORAS5 source ({src_dx:.3f}° × {src_dy:.3f}°).  "
            "Upscaling invents spurious fine-scale structure; "
            "consider a coarser target grid (≥ 0.25°).",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Public: regrid_to_model
# ---------------------------------------------------------------------------

def regrid_to_model(
    raw:    dict[str, Optional[np.ndarray]],
    grid:   OceanGrid,
    T_fill: float = 10.0,
    S_fill: float = 35.0,
) -> OceanState:
    """
    Interpolate raw ORAS5 fields onto the OceanJAX model grid.

    Parameters
    ----------
    raw    : dict returned by ``read_oras5()``.
    grid   : Target OceanGrid.
    T_fill : Temperature fallback [°C] for points outside ORAS5 coverage.
    S_fill : Salinity fallback [psu] for points outside ORAS5 coverage.

    Returns
    -------
    OceanState with masks applied, ready for ``step()`` / ``run()``.

    Notes
    -----
    Array layout: ORAS5 stores data as ``(Nz, Ny, Nx)``; OceanJAX uses
    ``(Nx, Ny, Nz)``.  Transposition is applied internally.

    Vertical extrapolation:
      T, S — nearest-neighbour (deepest valid ORAS5 value).
      u, v — zero below ORAS5 deepest level.
    """
    src_lon   = raw["lon"]
    src_lat   = raw["lat"]
    src_depth = raw["depth"]

    tgt_lon   = np.asarray(grid.lon_c)
    tgt_lat   = np.asarray(grid.lat_c)
    tgt_depth = np.asarray(grid.z_c)

    _check_resolution(src_lon, src_lat, tgt_lon, tgt_lat)

    if (tgt_lon.min() < src_lon.min() - 1e-6 or
            tgt_lon.max() > src_lon.max() + 1e-6 or
            tgt_lat.min() < src_lat.min() - 1e-6 or
            tgt_lat.max() > src_lat.max() + 1e-6):
        warnings.warn(
            "Target grid extends outside ORAS5 source domain.  "
            "Out-of-domain points will be filled by nearest-neighbour "
            "then constant fallback.",
            UserWarning,
            stacklevel=2,
        )

    kw3 = dict(src_lon=src_lon, src_lat=src_lat, src_depth=src_depth,
               tgt_lon=tgt_lon, tgt_lat=tgt_lat, tgt_depth=tgt_depth)
    kw2 = dict(src_lon=src_lon, src_lat=src_lat,
               tgt_lon=tgt_lon, tgt_lat=tgt_lat)

    Nz = len(tgt_depth)
    Ny = len(tgt_lat)
    Nx = len(tgt_lon)

    # Interpolate: each _interp_* returns (Nz_tgt, Ny_tgt, Nx_tgt)
    T_zyx = _interp_3d(raw["T"], **kw3, fill_value=T_fill)
    S_zyx = _interp_3d(raw["S"], **kw3, fill_value=S_fill)

    if raw["u"] is not None:
        u_zyx = _interp_3d(raw["u"], **kw3, fill_value=0.0)
        v_zyx = _interp_3d(raw["v"], **kw3, fill_value=0.0)
        # Zero out levels that lie below the deepest ORAS5 level.
        # Nearest-neighbour would otherwise extrapolate bottom velocities
        # indefinitely downward, which is physically wrong.
        below_src = tgt_depth > src_depth.max()
        if np.any(below_src):
            u_zyx[below_src] = 0.0
            v_zyx[below_src] = 0.0
    else:
        u_zyx = np.zeros((Nz, Ny, Nx), dtype=np.float32)
        v_zyx = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    eta_yx  = (_interp_2d(raw["eta"], **kw2, fill_value=0.0)
               if raw["eta"] is not None
               else np.zeros((Ny, Nx), dtype=np.float32))

    # Transpose (Nz, Ny, Nx) → (Nx, Ny, Nz) for OceanJAX convention
    T   = T_zyx.transpose(2, 1, 0)
    S   = S_zyx.transpose(2, 1, 0)
    u   = u_zyx.transpose(2, 1, 0)
    v   = v_zyx.transpose(2, 1, 0)
    eta = eta_yx.T                    # (Ny, Nx) → (Nx, Ny)

    return create_from_arrays(grid, u=u, v=v, T=T, S=S, eta=eta)


# ---------------------------------------------------------------------------
# Public: load_oras5
# ---------------------------------------------------------------------------

def load_oras5(
    path:       str | Path,
    grid:       OceanGrid,
    time_index: int   = 0,
    T_fill:     float = 10.0,
    S_fill:     float = 35.0,
) -> OceanState:
    """
    Convenience wrapper: read and regrid in one call.

    Equivalent to::

        regrid_to_model(read_oras5(path, time_index), grid, T_fill, S_fill)

    Parameters
    ----------
    path       : Copernicus Marine ORAS5 NetCDF file.
    grid       : Target OceanGrid.
    time_index : Time slice index within the file (default 0).
    T_fill     : Fallback temperature [°C] for uncovered ocean points.
    S_fill     : Fallback salinity [psu] for uncovered ocean points.

    Returns
    -------
    OceanState ready to pass to ``step()`` or ``run()``.
    """
    return regrid_to_model(
        read_oras5(path, time_index),
        grid,
        T_fill=T_fill,
        S_fill=S_fill,
    )
