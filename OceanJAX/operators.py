"""
OceanJAX Operators Module
=========================
Pure JAX functions for finite-difference operators on the Arakawa C-grid.

Conventions
-----------
- All operators return arrays of the same shape as the *output* staggering.
- Every flux at a face is gated by the mask of that face (mask_u, mask_v, mask_w)
  so that no stencil or flux crosses a land cell, the seafloor, or a dry cell.
  Scalar outputs at tracer centers are additionally zeroed by mask_c.
- Boundary conditions:
    x (zonal):      periodic via jnp.roll
    y (meridional):
        north face (j = Ny-1): zero-gradient Neumann — phi[Ny] := phi[Ny-1]
        south face (j = 0):    interior stencil applied; no explicit south BC
        south wall flux:       Fv_south = 0 at j = 0 (wall, no flux)
    z (vertical):   zero-flux enforced at k = 0 (surface) and k = Nz (bottom)
                    in grad_z; the surface kinematic BC (w = deta/dt) is set by
                    the dynamics layer, not here.
- For 2-D fields (e.g. SSH eta, shape (Nx, Ny)) the surface level (k=0) of
  the relevant 3-D mask is used.
- All functions are jit-compilable and jax.grad-compatible; no control flow
  on array values.
"""

from __future__ import annotations

import jax.numpy as jnp

from OceanJAX.grid import OceanGrid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask2d(mask3d: jnp.ndarray) -> jnp.ndarray:
    """Return the k=0 slice of a 3-D mask for use with 2-D (surface) fields."""
    return mask3d[:, :, 0]


def _expand(a: jnp.ndarray, is_3d: bool) -> jnp.ndarray:
    """Append a trailing size-1 axis for broadcasting against a 3-D field."""
    return a[:, :, jnp.newaxis] if is_3d else a


# ---------------------------------------------------------------------------
# Gradient operators
# ---------------------------------------------------------------------------

def grad_x(phi: jnp.ndarray, grid: OceanGrid) -> jnp.ndarray:
    """
    Zonal finite difference of tracer phi -> u-points (east face i+1/2).

    d phi / dx  at the east face.
    Periodic BC in x: phi[Nx, j, k] = phi[0, j, k].
    Result zeroed at dry u-faces via mask_u.
    Output shape matches phi.
    """
    is_3d = phi.ndim == 3
    dphi = jnp.roll(phi, -1, axis=0) - phi
    if is_3d:
        return dphi / grid.dx_u[:, :, jnp.newaxis] * grid.mask_u
    else:
        return dphi / grid.dx_u * _mask2d(grid.mask_u)


def grad_y(phi: jnp.ndarray, grid: OceanGrid) -> jnp.ndarray:
    """
    Meridional finite difference of tracer phi -> v-points (north face j+1/2).

    d phi / dy  at the north face.
    North boundary (j = Ny-1): zero-gradient Neumann — phi[Ny] := phi[Ny-1].
    South boundary (j = 0):    interior stencil; no explicit BC applied.
    Result zeroed at dry v-faces via mask_v.
    Output shape matches phi.
    """
    is_3d = phi.ndim == 3
    phi_n = jnp.roll(phi, -1, axis=1)
    phi_n = phi_n.at[:, -1, ...].set(phi[:, -1, ...])   # north Neumann
    dphi  = phi_n - phi
    if is_3d:
        return dphi / grid.dy_v[:, :, jnp.newaxis] * grid.mask_v
    else:
        return dphi / grid.dy_v * _mask2d(grid.mask_v)


def grad_z(phi: jnp.ndarray, grid: OceanGrid) -> jnp.ndarray:
    """
    Vertical finite difference of tracer phi (Nx,Ny,Nz) -> w-points (Nx,Ny,Nz+1).

    d phi / dz  at the top face of each layer (positive downward).
    Interior faces (k=1..Nz-1) are gated by mask_w.
    Surface (k=0) and bottom (k=Nz) are set to zero here; the surface
    kinematic BC (w = deta/dt or w = 0) is imposed by the dynamics layer.
    """
    dphi_interior = phi[..., 1:] - phi[..., :-1]          # (Nx, Ny, Nz-1)
    dz_interior   = grid.dz_w[1:grid.Nz]                  # (Nz-1,)
    grad_interior = (dphi_interior / dz_interior) * grid.mask_w[..., 1:grid.Nz]

    zeros = jnp.zeros(phi.shape[:2] + (1,), dtype=phi.dtype)
    return jnp.concatenate([zeros, grad_interior, zeros], axis=-1)  # (Nx, Ny, Nz+1)


# ---------------------------------------------------------------------------
# Divergence operators
# ---------------------------------------------------------------------------

def div_h(u: jnp.ndarray, v: jnp.ndarray, grid: OceanGrid) -> jnp.ndarray:
    """
    Horizontal divergence of (u, v) in flux form at tracer cell centres.

    East-face flux  Fu = u * dy_c   (face height = tracer-cell meridional extent)
    North-face flux Fv = v * dx_v   (face width at v-point latitude, cos(lat_v) metric)
    Both fluxes are gated by their respective face masks.
    Output: (Nx, Ny, Nz), zeroed at dry tracer cells.
    """
    Fu    = u * grid.mask_u * grid.dy_c[:, :, jnp.newaxis]
    dFu   = Fu - jnp.roll(Fu, 1, axis=0)

    Fv      = v * grid.mask_v * grid.dx_v[:, :, jnp.newaxis]
    Fv_s    = jnp.concatenate(
        [jnp.zeros((grid.Nx, 1, grid.Nz), dtype=v.dtype), Fv[:, :-1, :]], axis=1
    )
    dFv     = Fv - Fv_s

    return ((dFu + dFv) / grid.area_c[:, :, jnp.newaxis]) * grid.mask_c


def div_z(w: jnp.ndarray, grid: OceanGrid) -> jnp.ndarray:
    """
    Vertical divergence of w at tracer cell centres.

    div_z = (w[k+1] - w[k]) / dz_c[k]
    w is gated by mask_w before differencing.
    Output: (Nx, Ny, Nz), zeroed at dry tracer cells.
    """
    w_masked = w * grid.mask_w
    return ((w_masked[..., 1:] - w_masked[..., :-1]) / grid.dz_c) * grid.mask_c


# ---------------------------------------------------------------------------
# Interpolation operators
# ---------------------------------------------------------------------------

def interp_c_to_u(phi: jnp.ndarray, grid: OceanGrid) -> jnp.ndarray:
    """
    Interpolate tracer (Nx,Ny,Nz) to u-point (east face) by arithmetic
    average of phi[i] and phi[i+1].  Periodic in x.
    Zeroed at dry u-faces via mask_u.
    """
    return 0.5 * (phi + jnp.roll(phi, -1, axis=0)) * grid.mask_u


def interp_c_to_v(phi: jnp.ndarray, grid: OceanGrid) -> jnp.ndarray:
    """
    Interpolate tracer (Nx,Ny,Nz) to v-point (north face) by averaging
    phi[j] and phi[j+1].  North boundary: phi[Ny] := phi[Ny-1].
    Zeroed at dry v-faces via mask_v.
    """
    phi_n = jnp.roll(phi, -1, axis=1)
    phi_n = phi_n.at[:, -1, ...].set(phi[:, -1, ...])
    return 0.5 * (phi + phi_n) * grid.mask_v


def interp_u_to_c(u: jnp.ndarray, grid: OceanGrid) -> jnp.ndarray:
    """
    Interpolate u-point (Nx,Ny,Nz) to tracer center by averaging
    u[i-1/2] and u[i+1/2].  Periodic in x.
    Zeroed at dry tracer cells via mask_c.
    """
    return 0.5 * (u + jnp.roll(u, 1, axis=0)) * grid.mask_c


def interp_v_to_c(v: jnp.ndarray, grid: OceanGrid) -> jnp.ndarray:
    """
    Interpolate v-point (Nx,Ny,Nz) to tracer center by averaging
    v[j-1/2] and v[j+1/2].  South wall: v_south = 0 at j = 0.
    Zeroed at dry tracer cells via mask_c.
    """
    v_s = jnp.concatenate(
        [jnp.zeros((grid.Nx, 1, grid.Nz), dtype=v.dtype), v[:, :-1, :]], axis=1
    )
    return 0.5 * (v + v_s) * grid.mask_c


def interp_c_to_w(phi: jnp.ndarray, grid: OceanGrid) -> jnp.ndarray:
    """
    Interpolate tracer (Nx,Ny,Nz) to w-points (Nx,Ny,Nz+1) by averaging
    adjacent cell centres.
    At the surface (k=0) and bottom (k=Nz): zero-gradient Neumann extrapolation
    — the nearest cell value is copied to the boundary face.
    Result gated by mask_w.
    """
    interior = 0.5 * (phi[..., :-1] + phi[..., 1:])       # (Nx, Ny, Nz-1)
    avg = jnp.concatenate([phi[..., :1], interior, phi[..., -1:]], axis=-1)
    return avg * grid.mask_w


def interp_w_to_c(w: jnp.ndarray, grid: OceanGrid) -> jnp.ndarray:
    """
    Interpolate w-point (Nx,Ny,Nz+1) to tracer center (Nx,Ny,Nz)
    by averaging the two bounding w-faces.
    Zeroed at dry tracer cells via mask_c.
    """
    return 0.5 * (w[..., :-1] + w[..., 1:]) * grid.mask_c


# ---------------------------------------------------------------------------
# Laplacian / biharmonic diffusion operators
# ---------------------------------------------------------------------------

def laplacian_h(phi: jnp.ndarray, grid: OceanGrid) -> jnp.ndarray:
    """
    Horizontal Laplacian of phi in flux form at tracer cell centres.

      lap = (Fx_e - Fx_w + Fy_n - Fy_s) / area

    where
      Fx_e[i]  = dy_c[i]  * (phi[i+1] - phi[i]) / dx_u[i]   east face, gated mask_u
      Fx_w[i]  = Fx_e[i-1]                                   west face
      Fy_n[j]  = dx_v[j]  * (phi[j+1] - phi[j]) / dy_v[j]   north face, gated mask_v
      Fy_s[j]  = Fy_n[j-1],  Fy_s = 0 at j = 0             south wall zero-flux

    North face uses dx_v (cos(lat_v) metric) to account for spherical geometry
    and variable resolution.  Output zeroed at dry tracer cells via mask_c.

    Input:  phi (Nx, Ny) or (Nx, Ny, Nz)
    Output: same shape as phi
    """
    is_3d = phi.ndim == 3

    dy_c = _expand(grid.dy_c, is_3d)
    dx_u = _expand(grid.dx_u, is_3d)
    dx_v = _expand(grid.dx_v, is_3d)
    dy_v = _expand(grid.dy_v, is_3d)
    area = _expand(grid.area_c, is_3d)

    if is_3d:
        mu, mv, mc = grid.mask_u, grid.mask_v, grid.mask_c
    else:
        mu = _mask2d(grid.mask_u)
        mv = _mask2d(grid.mask_v)
        mc = _mask2d(grid.mask_c)

    # East-face flux
    phi_e = jnp.roll(phi, -1, axis=0)
    Fx_e  = dy_c * (phi_e - phi) / dx_u * mu

    # West-face flux = east flux of the western neighbor (periodic)
    Fx_w  = jnp.roll(Fx_e, 1, axis=0)

    # North-face flux (north Neumann BC)
    phi_n = jnp.roll(phi, -1, axis=1)
    if is_3d:
        phi_n = phi_n.at[:, -1, :].set(phi[:, -1, :])
    else:
        phi_n = phi_n.at[:, -1].set(phi[:, -1])
    Fy_n  = dx_v * (phi_n - phi) / dy_v * mv

    # South-face flux = north flux of the southern neighbor; zero at j = 0
    if is_3d:
        Fy_s = jnp.concatenate(
            [jnp.zeros(phi.shape[:1] + (1,) + phi.shape[2:], dtype=phi.dtype),
             Fy_n[:, :-1, :]], axis=1
        )
    else:
        Fy_s = jnp.concatenate(
            [jnp.zeros((phi.shape[0], 1), dtype=phi.dtype), Fy_n[:, :-1]], axis=1
        )

    return ((Fx_e - Fx_w + Fy_n - Fy_s) / area) * mc


def biharmonic_h(phi: jnp.ndarray, grid: OceanGrid) -> jnp.ndarray:
    """
    Biharmonic (del^4) horizontal operator: Lap_h(Lap_h(phi)).
    Useful for scale-selective dissipation at higher resolutions.
    """
    return laplacian_h(laplacian_h(phi, grid), grid)