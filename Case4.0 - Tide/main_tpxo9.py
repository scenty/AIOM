import pyTMD
import datetime

# TPXO9 配置
model = 'TPXO8-atlas' #
model_files = r'c:\Users\liuy\Desktop\潮位预报\tpxo9'  # 包含 h_tpxo9.v1.nc 的目录
grid_file = None  # 不需要网格文件
# - Other Parameters
n_sec_x_hour = 60 * 60
n_sec_x_day = 24 * 60 * 60
t_date_00 = datetime.datetime(year=2022, month=1, day=1)
t_date_11 = datetime.datetime(year=2022, month=2, day=1)
pt_lat = 26
pt_lon = 120

# - Compute difference in time expressed in hours
delta_hours = int((t_date_11 - t_date_00).total_seconds() / n_sec_x_hour)

# - Calculate Number of days relative to Jan 1, 1992 (48622 MJD)
# - using datetime
t_jd_ref = datetime.datetime(year=1992, month=1, day=1, hour=0)
t_est_tide = [t_date_00 + datetime.timedelta(hours=t)
                for t in range(delta_hours)]

# - Compute Datetime Values for selected date.
delta_time = [(t - t_jd_ref).total_seconds() / n_sec_x_day for t in
                t_est_tide]

# 计算潮高
elevations = pyTMD.compute.tide_elevations(pt_lon, pt_lat, delta_time, MODEL=model,
        DIRECTORY=r"c:\Users\liuy\Desktop\潮位预报\tpxo8_atlas_compact_v1.tar\DATA",
        model_files=model_files,
        grid_file=grid_file,
        TYPE='time series', TIME='UTC',
        method='bilinear',
        extrapolate=False
    )

tide_uv = pyTMD.compute.tide_currents(x, y, delta_time, DIRECTORY=path_to_tide_models, MODEL='CATS2008', EPSG=3031,
                                          EPOCH=(2000, 1, 1, 0, 0, 0), TYPE='drift', TIME='GPS', METHOD='spline',
                                          FILL_VALUE=np.nan)