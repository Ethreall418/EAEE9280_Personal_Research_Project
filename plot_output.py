"""
plot_output.py
==============
Visualise OceanJAX experiment output (NetCDF produced by experiment.py).

Usage
-----
    python plot_output.py [output.nc] [--savedir figures]

Defaults:
    output file  : output_cold_full_forcing.nc  (most recent .nc if absent)
    save dir     : figures/

Figures produced
----------------
  fig1_timeseries.png   — Domain-mean / min / max of T, S, eta over time.
  fig2_surface_maps.png — SST, SSS, eta maps at t=0, mid-run, t=final.
  fig3_profiles.png     — Domain-mean vertical T and S profiles at three times.
  fig4_hov_sst.png      — Hovmoller: zonal-mean SST vs. time.

Ensemble support
----------------
When the NetCDF contains a "member" dimension, each panel shows the
ensemble mean; shading shows the member spread (±1 std) where applicable.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import netCDF4 as nc_lib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: str | Path) -> dict:
    """Load all variables from the NetCDF into numpy arrays."""
    ds = nc_lib.Dataset(path)
    data = {v: ds.variables[v][:] for v in ds.variables}
    data["_ensemble"] = "member" in ds.dimensions
    data["_path"] = str(path)
    ds.close()
    return data


def _savefig(fig, savedir: Path, name: str) -> None:
    savedir.mkdir(parents=True, exist_ok=True)
    out = savedir / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


def _title_suffix(data: dict) -> str:
    return Path(data["_path"]).stem


# ---------------------------------------------------------------------------
# Figure 1 — Time series
# ---------------------------------------------------------------------------

def fig_timeseries(data: dict, savedir: Path) -> None:
    """Domain-mean / min / max time series of T, S, eta."""
    time_days = np.array(data["time"]) / 86400.0   # s -> days

    # For ensemble: average over member axis (axis 1 after time)
    ensemble = data["_ensemble"]

    def stats(arr, spatial_axes):
        """Return (mean, min, max) over spatial axes, after ensemble-mean."""
        if ensemble:
            arr = arr.mean(axis=1)                   # ensemble mean
        mu  = arr.mean(axis=spatial_axes)
        mn  = arr.min(axis=spatial_axes)
        mx  = arr.max(axis=spatial_axes)
        return mu, mn, mx

    T_mu, T_mn, T_mx   = stats(np.array(data["T"]),   (1, 2, 3))
    S_mu, S_mn, S_mx   = stats(np.array(data["S"]),   (1, 2, 3))
    eta_mu, eta_mn, eta_mx = stats(np.array(data["eta"]), (1, 2))

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.suptitle(f"Time series — {_title_suffix(data)}", fontsize=11)

    colors = {"mean": "#1f77b4", "range": "#aec7e8"}

    for ax, mu, mn, mx, label, unit in zip(
        axes,
        [T_mu,   S_mu,   eta_mu],
        [T_mn,   S_mn,   eta_mn],
        [T_mx,   S_mx,   eta_mx],
        ["Temperature", "Salinity", "SSH"],
        ["°C",   "psu",  "m"],
    ):
        ax.fill_between(time_days, mn, mx, color=colors["range"],
                        alpha=0.5, label="min–max")
        ax.plot(time_days, mu, color=colors["mean"], lw=1.8, label="domain mean")
        ax.set_ylabel(f"{label} [{unit}]", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, lw=0.4, alpha=0.5)

    axes[-1].set_xlabel("Simulation time [days]", fontsize=9)
    fig.tight_layout()
    _savefig(fig, savedir, "fig1_timeseries.png")


# ---------------------------------------------------------------------------
# Figure 2 — Surface maps
# ---------------------------------------------------------------------------

def fig_surface_maps(data: dict, savedir: Path) -> None:
    """SST, SSS, eta maps at t=0, t=mid, t=final."""
    lon = np.array(data["x"])     # (Nx,) degrees east
    lat = np.array(data["y"])     # (Ny,) degrees north
    time_days = np.array(data["time"]) / 86400.0
    nt = len(time_days)

    tidx = [0, nt // 2, nt - 1]
    t_labels = [f"Day {time_days[i]:.0f}" for i in tidx]

    ensemble = data["_ensemble"]

    def surface_field(arr):
        """Extract surface (k=0), ensemble-mean, shape (nt, Nx, Ny)."""
        if ensemble:
            arr = arr.mean(axis=1)   # (nt, Nx, Ny, Nz) or (nt, Nx, Ny)
        if arr.ndim == 4:
            return arr[:, :, :, 0]   # surface layer k=0  →  (nt, Nx, Ny)
        return arr                   # eta already (nt, Nx, Ny)

    T_surf   = surface_field(np.array(data["T"]))
    S_surf   = surface_field(np.array(data["S"]))
    eta_surf = surface_field(np.array(data["eta"]))

    fields = [
        (T_surf,   "SST",  "°C",  "RdYlBu_r"),
        (S_surf,   "SSS",  "psu", "viridis"),
        (eta_surf, "SSH",  "m",   "coolwarm"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    fig.suptitle(f"Surface maps — {_title_suffix(data)}", fontsize=11)

    for row, (arr, fname, unit, cmap) in enumerate(fields):
        # Consistent colour scale across the three time snapshots
        vmin = np.percentile(arr[arr != 0], 2)  if arr[arr != 0].size else arr.min()
        vmax = np.percentile(arr[arr != 0], 98) if arr[arr != 0].size else arr.max()
        if fname == "SSH":          # symmetric around zero
            vlim = max(abs(vmin), abs(vmax))
            vmin, vmax = -vlim, vlim

        for col, (ti, tlab) in enumerate(zip(tidx, t_labels)):
            ax = axes[row, col]
            # arr shape: (nt, Nx, Ny);  pcolormesh expects (Ny, Nx) or (Nx, Ny) grid
            field_2d = arr[ti].T                         # (Ny, Nx) for lat-lon plot
            pcm = ax.pcolormesh(lon, lat, field_2d,
                                cmap=cmap, vmin=vmin, vmax=vmax,
                                shading="auto")
            ax.set_title(f"{fname}  {tlab}", fontsize=8)
            ax.set_xlabel("Lon", fontsize=7)
            ax.set_ylabel("Lat", fontsize=7)
            ax.tick_params(labelsize=7)
            fig.colorbar(pcm, ax=ax, label=unit, pad=0.02, fraction=0.046)

    fig.tight_layout()
    _savefig(fig, savedir, "fig2_surface_maps.png")


# ---------------------------------------------------------------------------
# Figure 3 — Vertical profiles
# ---------------------------------------------------------------------------

def fig_profiles(data: dict, savedir: Path) -> None:
    """Domain-mean T and S vertical profiles at t=0, mid, final."""
    z   = -np.array(data["z"])   # depth positive downward for y-axis
    time_days = np.array(data["time"]) / 86400.0
    nt = len(time_days)

    ensemble = data["_ensemble"]

    def domain_mean_profile(arr):
        """(nt, [B,] Nx, Ny, Nz) → (nt, Nz) domain-mean profile."""
        if ensemble:
            arr = arr.mean(axis=1)       # ensemble mean: (nt, Nx, Ny, Nz)
        return arr.mean(axis=(1, 2))     # spatial mean: (nt, Nz)

    T_prof = domain_mean_profile(np.array(data["T"]))
    S_prof = domain_mean_profile(np.array(data["S"]))

    tidx   = [0, nt // 2, nt - 1]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = [f"Day {time_days[i]:.0f}" for i in tidx]

    fig, (ax_T, ax_S) = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
    fig.suptitle(f"Vertical profiles (domain mean) — {_title_suffix(data)}", fontsize=11)

    for ti, col, lab in zip(tidx, colors, labels):
        ax_T.plot(T_prof[ti], z, color=col, lw=1.8, label=lab)
        ax_S.plot(S_prof[ti], z, color=col, lw=1.8, label=lab)

    ax_T.set_xlabel("Temperature [°C]", fontsize=9)
    ax_T.set_ylabel("Depth [m]", fontsize=9)
    ax_T.invert_yaxis()
    ax_T.legend(fontsize=8)
    ax_T.grid(True, lw=0.4, alpha=0.5)

    ax_S.set_xlabel("Salinity [psu]", fontsize=9)
    ax_S.legend(fontsize=8)
    ax_S.grid(True, lw=0.4, alpha=0.5)

    fig.tight_layout()
    _savefig(fig, savedir, "fig3_profiles.png")


# ---------------------------------------------------------------------------
# Figure 4 — Hovmoller: zonal-mean SST vs time
# ---------------------------------------------------------------------------

def fig_hovmoller(data: dict, savedir: Path) -> None:
    """Hovmoller diagram: zonal-mean SST (lat vs time)."""
    lat       = np.array(data["y"])
    time_days = np.array(data["time"]) / 86400.0
    ensemble  = data["_ensemble"]

    T_arr = np.array(data["T"])
    if ensemble:
        T_arr = T_arr.mean(axis=1)          # (nt, Nx, Ny, Nz)
    sst = T_arr[:, :, :, 0]                 # surface layer: (nt, Nx, Ny)
    zonal_mean = sst.mean(axis=1)           # zonal mean: (nt, Ny)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"Hovmoller: zonal-mean SST — {_title_suffix(data)}", fontsize=11)

    pcm = ax.pcolormesh(time_days, lat, zonal_mean.T,
                        cmap="RdYlBu_r", shading="auto")
    fig.colorbar(pcm, ax=ax, label="SST [°C]", pad=0.02)
    ax.set_xlabel("Simulation time [days]", fontsize=9)
    ax.set_ylabel("Latitude [°]", fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.3, color="white")

    fig.tight_layout()
    _savefig(fig, savedir, "fig4_hov_sst.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise OceanJAX experiment NetCDF output."
    )
    parser.add_argument(
        "ncfile", nargs="?", default=None,
        help="Path to the NetCDF output file (default: output_cold_full_forcing.nc, "
             "or most recent .nc in the current directory).",
    )
    parser.add_argument(
        "--savedir", default="figures",
        help="Directory to save figures (default: figures/).",
    )
    args = parser.parse_args()

    # Resolve input file
    if args.ncfile is not None:
        ncpath = Path(args.ncfile)
    else:
        default = Path("output_cold_full_forcing.nc")
        if default.exists():
            ncpath = default
        else:
            candidates = sorted(Path(".").glob("output*.nc"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                print("ERROR: no output*.nc file found in current directory.",
                      file=sys.stderr)
                sys.exit(1)
            ncpath = candidates[0]
            print(f"No file specified; using most recent: {ncpath}")

    if not ncpath.exists():
        print(f"ERROR: file not found: {ncpath}", file=sys.stderr)
        sys.exit(1)

    savedir = Path(args.savedir)
    print(f"Reading: {ncpath}")

    data = _load(ncpath)
    nt   = len(data["time"])
    mode = "ensemble" if data["_ensemble"] else "single-run"
    print(f"  {nt} time records  mode={mode}")
    print(f"Saving figures to: {savedir}/")

    fig_timeseries(data, savedir)
    fig_surface_maps(data, savedir)
    fig_profiles(data, savedir)
    fig_hovmoller(data, savedir)

    print("Done.")


if __name__ == "__main__":
    main()
