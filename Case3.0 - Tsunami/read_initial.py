import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata

"""
极简版：从Excel读取数据并绘制带岸线的海啸初始场图

参数:
    file_path: Excel文件路径
    grid_resolution: 网格分辨率(度)，默认0.05
"""
grid_resolution=0.05
file_path = r"tsunami_deformation.xlsx"

df = pd.read_excel(file_path)
lon, lat, z = df.iloc[:,0], df.iloc[:,1], df.iloc[:,4]  # 假设前3列是lon,lat,z

# 2. 插值到网格
lon_grid = np.arange(lon.min(), lon.max()+grid_resolution, grid_resolution)
lat_grid = np.arange(lat.min(), lat.max()+grid_resolution, grid_resolution)
lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
z_grid = griddata((lon, lat), z, (lon_grid, lat_grid), method='linear', fill_value=0)

# 3. 创建带岸线的地图
fig = plt.figure(figsize=(10,8))
proj = ccrs.PlateCarree()  # 统一投影
ax = fig.add_subplot(111, projection=proj)

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.set_extent([100, 180, -50, 50])
# 4. 绘制数据
contour = ax.contourf(lon_grid, lat_grid, z_grid, levels=20, cmap='coolwarm')
plt.colorbar(contour, label='Vertical Displacement (m)')
plt.title('Tsunami Initial Field with Coastline')

import scipy.io as sio
DART = sio.loadmat('dart_data.mat')

# 使用统一的投影绘制浮标位置

for ii in range(len(DART['buoy_ids'])):
    if DART['lon'][0,ii]<180:
        ax.plot(DART['lon'][0,ii], DART['lat'][0,ii], 'ro')
        ax.text(DART['lon'][0,ii], DART['lat'][0,ii], DART['buoy_ids'][ii], 
                transform=proj, fontsize=8, color='black')

# 设置地图范围
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

plt.show()