import xarray as xr

files = [
    "votemper_control_monthly_highres_3D_202601_OPER_v0.1.nc",
    "vosaline_control_monthly_highres_3D_202601_OPER_v0.1.nc",
    "vozocrtx_control_monthly_highres_3D_202601_OPER_v0.1.nc",
    "vomecrty_control_monthly_highres_3D_202601_OPER_v0.1.nc",
    "sossheig_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "oras5_2026_01_native_merged.nc"
]

for f in files:
    ds = xr.open_dataset(f)
    print("\n====", f, "====")
    print(ds)
    print("data_vars:", list(ds.data_vars))
    print("coords:", list(ds.coords))