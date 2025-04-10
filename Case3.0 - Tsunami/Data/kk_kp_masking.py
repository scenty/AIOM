# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 21:23:18 2025

@author: Scenty
"""

from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.vectorized import contains
import numpy as np
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr


def create_land_mask(lon_grid, lat_grid, shoreline_file=None, resolution='50m'):
    """
    创建陆地掩膜（True表示陆地，False表示海洋），使用向量化方法加速。
    
    参数:
        lon_grid: 经度网格 (2D array)
        lat_grid: 纬度网格 (2D array)
        shoreline_file: 自定义岸线 Shapefile 路径（可选）
        resolution: Natural Earth 数据分辨率 ('10m', '50m', '110m')
    
    返回:
        mask: 2D 布尔数组（True表示陆地）
    """
    # 获取岸线几何数据
    if shoreline_file:
        # 从本地Shapefile读取（需要合适的包，如 fiona 或 geopandas）
        from fiona import collection as fiona_collection
        with fiona_collection(shoreline_file, 'r') as shp:
            geometries = [feature["geometry"] for feature in shp]
        # 注意：可能需要将字典转换为 shapely 对象
        from shapely.geometry import shape
        geometries = [shape(geom) for geom in geometries]
    else:
        # 使用Cartopy内置岸线数据
        land_feature = cfeature.NaturalEarthFeature(
            'physical', 'land', resolution,
            edgecolor='face', facecolor=cfeature.COLORS['land'])
        geometries = list(land_feature.geometries())
    
    # 修复并拆分几何图形
    valid_geoms = []
    for geom in geometries:
        valid_geom = geom  # 如果需要修复，可以调用 make_valid(geom)（需额外包支持）
        if valid_geom.geom_type == 'MultiPolygon':
            valid_geoms.extend(list(valid_geom.geoms))
        elif valid_geom.geom_type == 'Polygon':
            valid_geoms.append(valid_geom)
    
    # 合并所有多边形为一个单一几何（向量化操作要求一个 geometry 对象）
    land_polygons = unary_union(valid_geoms)
    
    # 使用向量化方法对整个网格进行判断（效率高得多）
    mask = contains(land_polygons, lon_grid, lat_grid)
    
    return mask

# 示例使用流程（注意调整数据加载部分以匹配你的环境）
if __name__ == "__main__":
    # 1. 打开原始 NetCDF 文件
    input_file = r"ETOPO1_Ice_c_gdal.nc"
    ds = xr.open_dataset(input_file)

    # 检查lat坐标的前几个数值（确保坐标顺序是否符合要求）
    print("lat坐标：", ds['lat'].values[:5])

    # 2. 根据经纬度裁剪感兴趣区域
    
    Lon_min = 120
    Lon_max = 300
    Lat_min = -60
    Lat_max = 60
    resolution = 1/60
    # 这里假设经度范围为 130°E 至 180°E，纬度范围为 -20° 至 60°（请根据实际数据情况确保lat顺序正确）
    ds_sub = ds.sel(lon=slice(Lon_min, Lon_max), lat=slice(Lat_min, Lat_max))

    # 3. 构造新的经纬度数组，步长均为 0.25°
    # 注意：使用 np.arange 时，终值可能不被包含，为了确保包含上界，可以设置终点为上界 + 分辨率
    new_lon = np.arange(Lon_min, Lon_max + resolution, resolution)
    new_lat = np.arange(Lat_min, Lat_max + resolution, resolution)

    # 4. 对子集数据进行插值处理（使用线性插值）
    ds_interp = ds_sub.interp(lon=new_lon, lat=new_lat, method="linear")

    # 5. 保存插值后的数据到新的 NetCDF 文件
    output_interp_file = r"ETOPO1_tsunami_large.nc"
    ds_interp.to_netcdf(output_interp_file)
    print(f"插值后的数据已保存到: {output_interp_file}")

    # 6. 绘制插值后的数据
    # 这里假设变量名为 'z'，请根据你的实际变量名称进行调整
    plt.figure(figsize=(8, 12))
    ds_interp['z'].plot(cmap='viridis', robust=True)
    plt.xlabel('Longitude (°E)')
    plt.ylabel('Latitude (°N)')
    plt.title('Subset of ETOPO1 Ice Surface Elevation with 0.25° Resolution')
    plt.show()

    # 1. 打开 tsunami_initial.nc 文件
    input_file = r"tsunami_initial.nc"  # 请确保文件路径正确
    ds = xr.open_dataset(input_file)

    # 检查纬度坐标顺序（打印前几个值）
    print("lat坐标：", ds['lat'].values[:5])

    # 2. 根据经纬度范围裁剪数据（例如经度 130°E 至 180°E，纬度 -20° 至 60°N）
    # 注意：如果实际的坐标顺序与这里假设的不一致（如 lat 为降序），需要调整 slice 参数
    ds_sub = ds.sel(lon=slice(130, 180), lat=slice(-20, 60))

    # 4. 对裁剪后的数据进行线性插值
    ds_interp = ds_sub.interp(lon=new_lon, lat=new_lat, method="linear")
    # 5. 用 fillna(0) 将插值后所有的空白值（NaN）填成 0
    ds_interp_filled = ds_interp.fillna(0)
    # 5. 保存插值后的数据到新的 NetCDF 文件
    output_file = r"tsunami_initial_interp.nc"
    ds_interp_filled.to_netcdf(output_file)
    print(f"插值后的数据已保存到: {output_file}")

    # 6. 绘制插值后的数据（假设插值变量名为 'z'，请根据实际情况调整）
    plt.figure(figsize=(8, 12))
    ds_interp_filled['z'].plot(cmap='viridis', vmin=-1.5, vmax=1.5)

    plt.xlabel('Longitude (°E)')
    plt.ylabel('Latitude (°N)')
    plt.title('Tsunami Initial Condition Interpolated at 0.25° Resolution')
    plt.show()
    
    # 1. 加载数据（这里以ETOPO1_Ice_c_gdal.nc为例）
    fname = r'ETOPO1_Ice_c_gdal_subset_interp_large.nc'
    nc_data = nc.Dataset(fname)
    lon = nc_data['lon'][:]
    lat = nc_data['lat'][:]
    zeta = nc_data['z'][:].squeeze()
    # 构造网格（注意：确保网格顺序和转换方式匹配你的数据）
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # 2. 创建掩膜
    mask = create_land_mask(lon_grid, lat_grid, resolution='50m')
    
    # 3. 保存掩膜结果到 NetCDF
    mask_ds = xr.Dataset(
        {'land_mask': (('lat', 'lon'), mask)},
        coords={'lat': lat, 'lon': lon}
    )
    mask_ds.to_netcdf('land_mask.nc')
    print("land_mask.nc 文件已生成。")
         
    # 打开 land_mask.nc 文件
    ds = xr.open_dataset('land_mask.nc')
    mask = ds['land_mask']
    
    # 打印基本信息（可选）
    print(ds)
    print("掩膜数据维度：", mask.shape)
    
    # 绘图
    plt.figure(figsize=(8,12))
    # 使用 pcolormesh 绘制掩膜图，灰度色图：陆地（值为1）和海洋（值为0）
    plt.pcolormesh(ds['lon'], ds['lat'], mask, cmap='gray', shading='auto')
    plt.colorbar(label='Land Mask (1: Land, 0: Sea)')
    plt.xlabel('Longitude (°E)')
    plt.ylabel('Latitude (°N)')
    plt.title('Japan Region Land Mask')
    plt.show()
    
    # 绘制 zeta 图（利用 pcolormesh 绘制二维数据）
    plt.figure(figsize=(8,12))
    plt.pcolormesh(lon, lat, zeta, cmap='viridis', shading='auto')
    plt.colorbar(label='Elevation (m)')
    plt.xlabel('Longitude (°E)')
    plt.ylabel('Latitude (°N)')
    plt.title('ETOPO1 Ice Surface Elevation (zeta)')
    plt.show()
    
    
    
    
