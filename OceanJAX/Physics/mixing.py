"""
OceanJAX Physics – Mixing Module
==================================
Vertical and horizontal mixing parameterisations.

Responsibility contract
-----------------------
Every function returns either a **tendency** [tracer s⁻¹ or m s⁻²] or a
**diffusivity field** [m² s⁻¹].  No time integration is performed here.

Vertical diffusion is treated **implicitly**: the time stepper calls
``implicit_vertical_mix`` which solves the tridiagonal system

  (I - dt * L_v) phi^{n+1} = phi^n + dt * explicit_tend

where L_v is the vertical diffusion operator, and returns phi^{n+1}
directly.  This avoids the severe stability constraint that would arise
from an explicit vertical diffusion step.

Horizontal viscosity and diffusion are treated **explicitly** and return
tendencies that the time stepper adds before advancing.

Contents
--------
thomas_algorithm          – differentiable tridiagonal solver via lax.scan
implicit_vertical_mix     – implicit vertical diffusion for tracers
implicit_vertical_visc    – implicit vertical diffusion for velocities (u or v)
horizontal_viscosity      – Laplacian viscosity tendency for u / v
richardson_number         – local gradient Richardson number
kpp_diffusivity           – simplified KPP vertical diffusivity
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from OceanJAX.grid import OceanGrid


# ---------------------------------------------------------------------------
# Tridiagonal solver (Thomas algorithm) – differentiable via lax.scan
# ---------------------------------------------------------------------------

def thomas_algorithm(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray,
) -> jnp.ndarray:
    """
    Solve the tridiagonal system  A x = d  using the Thomas algorithm.

    The system has the form:

      b[0] x[0] + c[0] x[1]                          = d[0]
      a[k] x[k-1] + b[k] x[k] + c[k] x[k+1]         = d[k]   1 ≤ k ≤ N-2
                    a[N-1] x[N-2] + b[N-1] x[N-1]    = d[N-1]

    Args:
        a : (N,) lower diagonal  (a[0] is unused)
        b : (N,) main  diagonal
        c : (N,) upper diagonal  (c[N-1] is unused)
        d : (N,) right-hand side

    Returns:
        x : (N,) solution

    Implementation uses ``jax.lax.scan`` so the solver is fully
    differentiable via reverse-mode AD and JIT-compilable.
    No Python loops over array values.
    """
    N = b.shape[0]

    # ---- Forward sweep: eliminate lower diagonal ---------------------------
    def fwd_step(carry, k):
        b_prev, d_prev = carry          # modified b and d from previous row
        w   = a[k] / b_prev            # elimination factor
        b_k = b[k] - w * c[k - 1]     # modified main diagonal
        d_k = d[k] - w * d_prev       # modified RHS
        return (b_k, d_k), (b_k, d_k)

    # Initialise with row 0
    init    = (b[0], d[0])
    # Scan over rows 1..N-1
    _, (b_mod, d_mod) = jax.lax.scan(fwd_step, init, jnp.arange(1, N))

    # Concatenate row 0 back
    b_all = jnp.concatenate([b[:1], b_mod])   # (N,)
    d_all = jnp.concatenate([d[:1], d_mod])   # (N,)

    # ---- Back substitution -------------------------------------------------
    def bwd_step(x_next, k):
        x_k = (d_all[k] - c[k] * x_next) / b_all[k]
        return x_k, x_k

    # Initialise with last row
    x_last = d_all[-1] / b_all[-1]
    _, x_interior = jax.lax.scan(
        bwd_step, x_last, jnp.arange(N - 2, -1, -1)
    )
    # x_interior is reversed (N-1 entries); append x_last and flip
    x = jnp.concatenate([x_interior[::-1], jnp.array([x_last])])
    return x


def _build_tridiag_implicit(kappa: jnp.ndarray, dz_c: jnp.ndarray,
                             dz_w: jnp.ndarray, dt: float,
                             mask_w_col: jnp.ndarray) -> tuple:
    """
    Build the tridiagonal coefficients for implicit vertical diffusion
    of a single water column.

    Discretisation of  -d/dz(kappa * dC/dz) at cell centres:

      flux_k   = kappa[k] / dz_w[k]    (flux coefficient at w-face k)

    The implicit system for column update C^{n+1}:

      C^{n+1}[k] - dt * (flux_{k+1} * C^{n+1}[k+1]
                        - (flux_k + flux_{k+1}) * C^{n+1}[k]
                        + flux_k * C^{n+1}[k-1]) / dz_c[k]
      = C^n[k]

    Boundary conditions (via mask_w_col):
      surface face k=0  : flux = 0 (Neumann, surface forcing handled separately)
      bottom  face k=Nz : flux = 0 (Neumann, no-flux seafloor)

    Args:
        kappa      : (Nz+1,) diffusivity at w-faces [m² s⁻¹]
        dz_c       : (Nz,)   cell thicknesses [m]
        dz_w       : (Nz+1,) distances between cell centres [m]
        dt         : timestep [s]
        mask_w_col : (Nz+1,) w-face mask for this column

    Returns:
        (a, b, c) tridiagonal coefficients, each (Nz,)
    """
    Nz = dz_c.shape[0]

    # Flux coefficients at each w-face [s⁻¹ equivalent]
    # Safe division: dz_w > 0 everywhere except possibly boundary
    safe_dz_w = jnp.where(dz_w > 0, dz_w, 1.0)
    flux = kappa * mask_w_col / safe_dz_w          # (Nz+1,)

    flux_top = flux[:Nz]    # face k   (top of cell k)
    flux_bot = flux[1:]     # face k+1 (bottom of cell k)

    # Lower diagonal: a[k] = -dt * flux_top[k] / dz_c[k]   (k=1..Nz-1)
    a = -dt * flux_top / dz_c                      # (Nz,)  a[0] unused
    # Upper diagonal: c[k] = -dt * flux_bot[k] / dz_c[k]   (k=0..Nz-2)
    c = -dt * flux_bot / dz_c                      # (Nz,)  c[Nz-1] unused
    # Main diagonal: b[k] = 1 - a[k] - c[k]
    b = 1.0 - a - c                                # (Nz,)

    return a, b, c


# ---------------------------------------------------------------------------
# Implicit vertical diffusion for tracers
# ---------------------------------------------------------------------------

def implicit_vertical_mix(
    phi:   jnp.ndarray,
    kappa: jnp.ndarray,
    dt:    float,
    grid:  OceanGrid,
    rhs_explicit: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Implicitly mix a tracer field in the vertical direction.

    Solves column-by-column:

      (I - dt * L_v) phi^{n+1} = phi^n + dt * rhs_explicit

    where L_v is the vertical diffusion operator.  A separate call to
    ``thomas_algorithm`` is made for each (i, j) column via ``jax.vmap``.

    Args:
        phi          : (Nx, Ny, Nz)    tracer field at time n
        kappa        : (Nx, Ny, Nz+1)  vertical diffusivity at w-faces [m² s⁻¹]
        dt           : timestep [s]
        grid         : OceanGrid
        rhs_explicit : (Nx, Ny, Nz) or None
                       Explicit tendency already accumulated for this timestep.
                       If None, treated as zero.

    Returns:
        phi^{n+1} : (Nx, Ny, Nz), masked to zero on dry cells
    """
    if rhs_explicit is None:
        rhs = phi
    else:
        rhs = phi + dt * rhs_explicit

    def solve_column(phi_col, kappa_col, rhs_col, mask_w_col, mask_c_col):
        """Solve one (i,j) column. All inputs are 1-D in z."""
        a, b, c = _build_tridiag_implicit(
            kappa_col, grid.dz_c, grid.dz_w, dt, mask_w_col
        )
        # For dry cells, the system degenerates; keep phi = 0 there.
        # The mask on the diagonal (b=1 for dry cells, a=c=0) achieves this
        # naturally since rhs = 0 for dry cells.
        phi_new = thomas_algorithm(a, b, c, rhs_col * mask_c_col)
        return phi_new * mask_c_col

    # vmap over (i, j) simultaneously by flattening the horizontal dims
    Nx, Ny, Nz = phi.shape
    phi_2d    = phi.reshape(Nx * Ny, Nz)
    kappa_2d  = kappa.reshape(Nx * Ny, Nz + 1)
    rhs_2d    = rhs.reshape(Nx * Ny, Nz)
    mask_w_2d = grid.mask_w.reshape(Nx * Ny, Nz + 1)
    mask_c_2d = grid.mask_c.reshape(Nx * Ny, Nz)

    phi_new_2d = jax.vmap(solve_column)(
        phi_2d, kappa_2d, rhs_2d, mask_w_2d, mask_c_2d
    )
    return phi_new_2d.reshape(Nx, Ny, Nz)


# ---------------------------------------------------------------------------
# Implicit vertical diffusion for velocity (u or v)
# ---------------------------------------------------------------------------

def implicit_vertical_visc(
    vel:   jnp.ndarray,
    nu_v:  jnp.ndarray,
    dt:    float,
    grid:  OceanGrid,
    mask:  jnp.ndarray,
) -> jnp.ndarray:
    """
    Implicitly mix a horizontal velocity component in the vertical direction.

    Identical structure to ``implicit_vertical_mix`` but uses the supplied
    velocity mask (mask_u or mask_v) instead of mask_c / mask_w.

    Args:
        vel   : (Nx, Ny, Nz)    velocity component at time n
        nu_v  : (Nx, Ny, Nz+1) vertical viscosity at w-faces [m² s⁻¹]
        dt    : timestep [s]
        grid  : OceanGrid
        mask  : (Nx, Ny, Nz)   mask for this velocity component

    Returns:
        vel^{n+1} : (Nx, Ny, Nz)
    """
    Nx, Ny, Nz = vel.shape

    def solve_column(vel_col, nu_col, mask_w_col, mask_col):
        a, b, c = _build_tridiag_implicit(
            nu_col, grid.dz_c, grid.dz_w, dt, mask_w_col
        )
        return thomas_algorithm(a, b, c, vel_col * mask_col) * mask_col

    vel_2d    = vel.reshape(Nx * Ny, Nz)
    nu_2d     = nu_v.reshape(Nx * Ny, Nz + 1)
    mask_w_2d = grid.mask_w.reshape(Nx * Ny, Nz + 1)
    mask_2d   = mask.reshape(Nx * Ny, Nz)

    vel_new_2d = jax.vmap(solve_column)(vel_2d, nu_2d, mask_w_2d, mask_2d)
    return vel_new_2d.reshape(Nx, Ny, Nz)


# ---------------------------------------------------------------------------
# Horizontal viscosity (explicit Laplacian)
# ---------------------------------------------------------------------------

def horizontal_viscosity(
    u:     jnp.ndarray,
    v:     jnp.ndarray,
    nu_h:  float | jnp.ndarray,
    grid:  OceanGrid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Explicit horizontal Laplacian viscosity tendencies for (u, v).

    Each velocity component is treated as a scalar field and passed through
    the same flux-form Laplacian used for tracers, with the appropriate
    face mask.

    For u (on east faces, mask_u):
      nu_h * lap_h(u)  at u-points

    For v (on north faces, mask_v):
      nu_h * lap_h(v)  at v-points

    Note: a geometrically consistent viscosity on the C-grid requires the
    full vector Laplacian, including off-diagonal stress terms, which are
    absent here.  This scalar approximation is standard in z-level models
    at moderate resolution.

    Args:
        u    : (Nx, Ny, Nz) zonal velocity
        v    : (Nx, Ny, Nz) meridional velocity
        nu_h : scalar or (Nx, Ny, Nz) horizontal viscosity [m² s⁻¹]
        grid : OceanGrid

    Returns:
        (du_dt_visc, dv_dt_visc) : each (Nx, Ny, Nz)
    """
    from OceanJAX.Physics.tracers import kappa_laplacian_h   # avoid circular at module level
    du = kappa_laplacian_h(u, nu_h, grid)
    dv = kappa_laplacian_h(v, nu_h, grid)
    return du, dv


# ---------------------------------------------------------------------------
# Richardson-number-based background diffusivity
# ---------------------------------------------------------------------------

def richardson_number(
    T:    jnp.ndarray,
    S:    jnp.ndarray,
    u:    jnp.ndarray,
    v:    jnp.ndarray,
    grid: OceanGrid,
    params,
) -> jnp.ndarray:
    """
    Gradient Richardson number at w-faces (Nx, Ny, Nz+1).

      Ri = N² / (du/dz)²

    where N² is the squared buoyancy frequency:

      N² = -(g / rho0) * drho/dz   (positive = stable)

    and the velocity shear squared:

      S² = (du/dz)² + (dv/dz)²

    Both are evaluated at w-faces using centred differences.
    A small floor on S² prevents division by zero.

    Args:
        T, S  : (Nx, Ny, Nz) temperature and salinity
        u, v  : (Nx, Ny, Nz) horizontal velocities
        grid  : OceanGrid
        params: ModelParams

    Returns:
        Ri : (Nx, Ny, Nz+1), clamped to [0, Ri_max]
    """
    from OceanJAX.Physics.dynamics import equation_of_state  # deferred

    rho = equation_of_state(T, S, params)   # (Nx, Ny, Nz)

    # Centred vertical differences at w-faces (same stencil as grad_z)
    # Interior faces k=1..Nz-1; boundaries set to 0
    def _diff_w(phi):
        interior = phi[..., 1:] - phi[..., :-1]   # (Nx, Ny, Nz-1)
        zeros    = jnp.zeros(phi.shape[:2] + (1,), dtype=phi.dtype)
        return jnp.concatenate([zeros, interior, zeros], axis=-1)  # (Nx, Ny, Nz+1)

    drho_dz = _diff_w(rho) / jnp.where(grid.dz_w > 0, grid.dz_w, 1.0)
    du_dz   = _diff_w(u)   / jnp.where(grid.dz_w > 0, grid.dz_w, 1.0)
    dv_dz   = _diff_w(v)   / jnp.where(grid.dz_w > 0, grid.dz_w, 1.0)

    n2 = -(params.g / params.rho0) * drho_dz   # (Nx, Ny, Nz+1)
    s2 = du_dz ** 2 + dv_dz ** 2               # shear squared

    ri = n2 / jnp.where(s2 > 1e-10, s2, 1e-10)
    ri = jnp.clip(ri, 0.0, 100.0)              # clamp to [0, Ri_max]
    return ri * grid.mask_w


# ---------------------------------------------------------------------------
# Simplified KPP vertical diffusivity
# ---------------------------------------------------------------------------

def kpp_diffusivity(
    T:     jnp.ndarray,
    S:     jnp.ndarray,
    u:     jnp.ndarray,
    v:     jnp.ndarray,
    grid:  OceanGrid,
    params,
    kappa_0:    float = 1e-5,
    kappa_conv: float = 1e-1,
    ri_crit:    float = 0.7,
) -> jnp.ndarray:
    """
    Simplified KPP-inspired vertical diffusivity for tracers.

    The scheme enhances the background diffusivity when the gradient
    Richardson number falls below a critical value and applies convective
    adjustment (large diffusivity) in statically unstable columns:

      kappa_v(k) = kappa_0
                 + kappa_conv * max(0, 1 - Ri / Ri_crit)²   if Ri < Ri_crit
                 + kappa_conv                                 if N² < 0

    This is a single-function approximation to the full KPP scheme
    (Large et al. 1994) without an explicit boundary-layer depth calculation.
    It is sufficient for stable multi-year integrations and can be replaced
    by an ML closure (see OceanJAX.ml.closure) or a more complete KPP
    implementation without changing the calling interface.

    Args:
        T, S        : (Nx, Ny, Nz)
        u, v        : (Nx, Ny, Nz)
        grid        : OceanGrid
        params      : ModelParams  (uses g, rho0, kappa_v as background)
        kappa_0     : background diffusivity [m² s⁻¹]  (default 1e-5)
        kappa_conv  : convective diffusivity [m² s⁻¹]  (default 0.1)
        ri_crit     : critical Richardson number        (default 0.7)

    Returns:
        kappa : (Nx, Ny, Nz+1) [m² s⁻¹], zeroed at dry w-faces
    """
    from OceanJAX.Physics.dynamics import equation_of_state  # deferred

    rho = equation_of_state(T, S, params)

    def _diff_w(phi):
        interior = phi[..., 1:] - phi[..., :-1]
        zeros    = jnp.zeros(phi.shape[:2] + (1,), dtype=phi.dtype)
        return jnp.concatenate([zeros, interior, zeros], axis=-1)

    safe_dz_w = jnp.where(grid.dz_w > 0, grid.dz_w, 1.0)
    drho_dz   = _diff_w(rho) / safe_dz_w
    du_dz     = _diff_w(u)   / safe_dz_w
    dv_dz     = _diff_w(v)   / safe_dz_w

    n2  = -(params.g / params.rho0) * drho_dz
    s2  = du_dz ** 2 + dv_dz ** 2
    ri  = n2 / jnp.where(s2 > 1e-10, s2, 1e-10)

    # Shear-driven enhancement: active when 0 <= Ri < ri_crit
    shear_factor = jnp.where(
        (ri >= 0) & (ri < ri_crit),
        (1.0 - ri / ri_crit) ** 2,
        0.0,
    )

    # Convective adjustment: active when N² < 0 (unstable column)
    conv_factor = jnp.where(n2 < 0, 1.0, 0.0)

    kappa = (kappa_0
             + kappa_conv * shear_factor
             + kappa_conv * conv_factor)

    return kappa * grid.mask_w
