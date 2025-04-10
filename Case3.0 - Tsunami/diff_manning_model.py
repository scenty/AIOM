import numpy as np
#import vis_tools
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#from J65 import *
from matplotlib.pyplot import *
import xarray as xr
import torch
import torch.nn.functional as F

from tool_train import ddx,ddy,rho2u,rho2v,v2rho,u2rho,dd
from tool_train import ududx_up,vdudy_up,udvdx_up,vdvdy_up
from All_params import Params_small,Params_large,Params_tsunami
from dynamics import mass_cartesian_torch,momentum_nonlinear_cartesian_torch

params = Params_tsunami()
device = torch.device('cpu')

dt = params.dt
NT = params.NT

x = torch.linspace(0, params.Lx, params.Nx + 1)
y = torch.linspace(0, params.Ly, params.Ny + 1)
X, Y = torch.meshgrid(x, y, indexing='ij')

M = torch.zeros((2, params.Nx, params.Ny+1))
N = torch.zeros((2, params.Nx+1, params.Ny))
# -------------------- 新增代码 --------------------


# 读取初始水位（tsunami_initial_interp.nc）——变量名假设为'z'
# 读取初始水位数据
ds_eta = xr.open_dataset('Data/tsunami_initial_interp.nc')
H_ini = torch.from_numpy(ds_eta['z'].values.astype(np.float64))
H_ini = H_ini.T  # 调整为 [51, 81]，与模型网格一致


# 填充 NaN 值为 0
H_ini = torch.nan_to_num(H_ini, nan=0.0)



# 读取 bathymetry（ETOPO1_Ice_c_gdal_subset_interp.nc）——变量名假设为'z'
# 读取 ETOPO1 底部高程数据（变量名假设为 'z'）
ds_Z = xr.open_dataset('Data/ETOPO1_Ice_c_gdal_subset_interp.nc')
Z = torch.from_numpy(ds_Z['z'].values.astype(np.float64))
Z = Z.T  # 调整为 [51, 81]
Z = -torch.nan_to_num(Z, nan=0.0)
Z[Z<params.dry_limit] = params.dry_limit


# 读取陆域掩模（land_mask.nc）——变量名假设为'mask'，1为陆，0为海
ds_land = xr.open_dataset('Data/land_mask.nc')


land_mask = torch.from_numpy(ds_land['land_mask'].values.astype(np.float64)).T
# --------------------------------------------------

# 替换原来的初始化：
# 原来:
#    H = torch.zeros((2, params.Nx+1, params.Ny+1))
#    Z = torch.ones((params.Nx+1, params.Ny+1)) * params.depth
# 修改后:
H = torch.zeros((2, params.Nx+1, params.Ny+1), dtype=torch.float64)
H[0] = H_ini  # 现在 H_ini 中不含 NaN


# H[0] = 1 * torch.exp(-((X-X.max()//2)**2 + (Y-Y.max()//2)**2) / (2 * 50000**2))
#
# Wx,Wy,Pa = generate_wind(X,Y,params)
# Ws = torch.sqrt(Wx**2 + Wy**2)
#
Wx = np.ones((NT,X.shape[0],X.shape[1]))*0 + 0
Wy = np.ones((NT,X.shape[0],X.shape[1]))*0 
Wx = torch.from_numpy(Wx)
Wy = torch.from_numpy(Wy)
Pa = Wx * 0 + 1000

eta_list = list()
u_list = list()
v_list = list()
Fx_list = list()
Fy_list = list()
sustr_list = list()
svstr_list = list()

anim_interval = 10
manning = torch.zeros_like(H_ini)
for itime in range(params.NT):
    print(f"nt = {itime} / {params.NT}")
    params.itime = itime
    #manning = model(current_input).squeeze(0).squeeze(0)
    H_update = mass_cartesian_torch(H, Z, M, N, params)
    
    # 应用陆域掩模：陆地（mask==1）处水位与动量置零
    H_update[1] = torch.where(land_mask == 1, torch.zeros_like(H_update[1]), H_update[1])
    
    M_update, N_update, Fx, Fy = momentum_nonlinear_cartesian_torch(H_update, Z, M, N, Wx[itime], Wy[itime], Pa[itime], params, manning)
    
    # 对 M（尺寸：(Nx, Ny+1)）使用 land_mask 的前 Nx 行：
    M_update[1] = torch.where(land_mask[:-1, :] == 1, torch.zeros_like(M_update[1]), M_update[1])
    
    # 对 N（尺寸：(Nx+1, Ny)）使用 land_mask 的前 Ny 列：
    N_update[1] = torch.where(land_mask[:, :-1] == 1, torch.zeros_like(N_update[1]), N_update[1])

    
    H[0] = H_update[1]
    M[0] = M_update[1]
    N[0] = N_update[1]
    eta_list.append(dd(H_update[1]))
    u_list.append(dd(M_update[1]))
    v_list.append(dd(N_update[1]))
    Fx_list.append(dd(Fx))
    Fy_list.append(dd(Fy))

    
    if itime % anim_interval == 0:
        # 从 M_update 与 N_update 计算 u, v 分量
        u = rho2u(M_update[1])
        v = rho2v(N_update[1])
        # 按相邻两列/两行平均，求得 cell center 对应的速度
        u_center = 0.5 * (u[:, :-1] + u[:, 1:])  # shape: (Nx, Ny)
        v_center = 0.5 * (v[:-1, :] + v[1:, :])    # shape: (Nx, Ny) ——理论上应相同
    
        # 若 u_center 与 v_center 尺寸不一致，则取尺寸交集
        min_rows = min(u_center.shape[0], v_center.shape[0])
        min_cols = min(u_center.shape[1], v_center.shape[1])
        u_center = u_center[:min_rows, :min_cols]
        v_center = v_center[:min_rows, :min_cols]
        mag = np.sqrt(u_center**2 + v_center**2)
    
        # 计算 cell center 坐标（基于原始 X, Y 定义在 rho 点，其尺寸为 (Nx+1, Ny+1)）
        Xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[:-1, 1:] + X[1:, 1:])
        Yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[:-1, 1:] + Y[1:, 1:])
        # 裁剪 cell center 坐标以匹配速度数据尺寸
        Xc = Xc[:min_rows, :min_cols]
        Yc = Yc[:min_rows, :min_cols]
    
        plt.figure(figsize=(8, 12))
        # 绘制阶段（η）数据，这里 η_list[-1] 的尺寸应与 X, Y 一致
        plt.pcolor(X, Y, eta_list[-1], vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
        plt.colorbar()
        # 绘制速度幅值的等高线，并显示 cell center 上的速度矢量
        # plt.contour(Xc, Yc, mag, colors='black')
        plt.quiver(Xc, Yc, u_center, v_center, scale=10000, color='k')
        
        plt.xlabel("x [m]", fontname="serif", fontsize=12)
        plt.ylabel("y [m]", fontname="serif", fontsize=12)
        plt.title("Stage at an instant in time: " + f"{itime*params.dt}" + " second")
        plt.axis('equal')
        plt.show()



v_array = np.array(v_list)
u_array = np.array(u_list)
eta = np.array(eta_list)
Fx_array = np.array(Fx_list)
Fy_array = np.array(Fy_list)
sustr_array = np.array(sustr_list)
svstr_array = np.array(svstr_list)
   
np.save('v_large_array.npy', v_array)
np.save('u_large_array.npy', u_array)
np.save('eta_large.npy', eta)
np.save('Fx_large_array.npy', Fx_array)
np.save('Fy_large_array.npy', Fy_array) 


   
# u_center = np.empty((u_array.shape[0], x.shape[0], y.shape[0]))
# u_center[:, 1:-1, :] = 0.5 * (u_array[:, :-1, :] + u_array[:, 1:, :])
# # 边界处理：这里可以选择外推或者简单复制
# u_center[:, 0, :] = u_array[:, 0, :]
# u_center[:, -1, :] = u_array[:, -1, :]

# # 对 v 在 y 方向进行外推或者插值
# v_center = np.empty((v_array.shape[0], x.shape[0], y.shape[0]))
# v_center[:, :, 1:-1] = 0.5 * (v_array[:, :, :-1] + v_array[:, :, 1:])
# v_center[:, :, 0] = v_array[:, :, 0]
# v_center[:, :, -1] = v_array[:, :, -1]
    
# 保存 eta 数据
ds_eta = xr.Dataset(
    {'eta': (['time', 'lat', 'lon'], np.array(eta_list))},
    coords={
        'lat': x,
        'lon': y,
        'time': np.arange(len(eta_list))
    }
)
ds_eta.to_netcdf('out_large_eta.nc')

# 保存 u 数据
ds_u = xr.Dataset(
    {'u': (['time', 'lat', 'lon'], np.array(u_array))},
    coords={
        'lat': x[:-1],
        'lon': y,
        'time': np.arange(len(u_array))
    }
)
ds_u.to_netcdf('out_large_u.nc')

# 保存 v 数据
ds_v = xr.Dataset(
    {'v': (['time', 'lat', 'lon'], np.array(v_array))},
    coords={
        'lat': x,
        'lon': y[:-1],
        'time': np.arange(len(v_array))
    }
)
ds_v.to_netcdf('out_large_v.nc')

np.save('manning_large_distribution.npy', params.manning.numpy())    