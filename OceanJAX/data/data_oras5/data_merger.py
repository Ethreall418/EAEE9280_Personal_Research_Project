import xarray as xr
from pathlib import Path

base = Path(r"C:\Users\EEEthreall\PycharmProjects\EAEE9280_Personal_Research_Project\OceanJAX\data\data_oras5")

ds_T = xr.open_dataset(base / "votemper_control_monthly_highres_3D_202601_OPER_v0.1.nc")
ds_S = xr.open_dataset(base / "vosaline_control_monthly_highres_3D_202601_OPER_v0.1.nc")
ds_U = xr.open_dataset(base / "vozocrtx_control_monthly_highres_3D_202601_OPER_v0.1.nc")
ds_V = xr.open_dataset(base / "vomecrty_control_monthly_highres_3D_202601_OPER_v0.1.nc")
ds_H = xr.open_dataset(base / "sossheig_control_monthly_highres_2D_202601_OPER_v0.1.nc")

# 去掉重复的 time_counter_bnds，避免 merge 冲突
for ds in [ds_S, ds_U, ds_V, ds_H]:
    if "time_counter_bnds" in ds:
        ds = ds.drop_vars("time_counter_bnds")

merged = xr.merge(
    [
        ds_T,
        ds_S.drop_vars("time_counter_bnds", errors="ignore"),
        ds_U.drop_vars("time_counter_bnds", errors="ignore"),
        ds_V.drop_vars("time_counter_bnds", errors="ignore"),
        ds_H.drop_vars("time_counter_bnds", errors="ignore"),
    ],
    compat="override"
)

print(merged)
print(list(merged.data_vars))
merged.to_netcdf(base / "oras5_2026_01_native_merged.nc")