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
import cartopy.crs as ccrs
import cartopy.feature as cfeature

params = Params_tsunami()
device = torch.device('cpu')

dt = params.dt
NT = params.NT
# 读取网格
ds_grid = xr.open_dataset('Data/tsunami_grid_1.0.nc')
x = torch.from_numpy(ds_grid['lon'][:].values)
y = torch.from_numpy(ds_grid['lat'][:].values)
X, Y = torch.meshgrid(x, y, indexing='ij')
# 地形
Z = torch.from_numpy(ds_grid['z'].values.astype(np.float64))
Z = Z.T  # 调整为 [51, 81]
Z = -torch.nan_to_num(Z, nan=0.0)
Z[Z<params.dry_limit] = params.dry_limit
land_mask = torch.from_numpy(ds_grid['mask'].values).T
# 初始场
H_ini = torch.from_numpy(ds_grid['zeta'].values.astype(np.float64))
H_ini = H_ini.T  # 调整为 [51, 81]，与模型网格一致
H_ini = torch.nan_to_num(H_ini, nan=0.0)
H = torch.zeros((2, params.Nx+1, params.Ny+1), dtype=torch.float64)
H[0] = H_ini  

M = torch.zeros((2, params.Nx, params.Ny+1))
N = torch.zeros((2, params.Nx+1, params.Ny))
# 驱动
#
# Wx,Wy,Pa = generate_wind(X,Y,params)
# Ws = torch.sqrt(Wx**2 + Wy**2)
#
Wx = np.ones((1,X.shape[0],X.shape[1]))*0 + 0
Wy = np.ones((1,X.shape[0],X.shape[1]))*0 
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

anim_interval = 1
manning = torch.zeros_like(H_ini)
for itime in range(params.NT):
    print(f"nt = {itime} / {params.NT}")
    params.itime = itime
    #manning = model(current_input).squeeze(0).squeeze(0)
    M_update, N_update, _ , _ = momentum_nonlinear_cartesian_torch(H, Z, M, N, Wx[0], Wy[0], Pa[0], params, manning)
    H_update = mass_cartesian_torch(H, Z, M_update, N_update, params)
    
    # 应用陆域掩模：陆地（mask==1）处水位与动量置零
    H_update[1] = torch.where(land_mask, torch.zeros_like(H_update[1]), H_update[1])
    M_update[1] = torch.where(land_mask[:-1, :], torch.zeros_like(M_update[1]), M_update[1])
    N_update[1] = torch.where(land_mask[:, :-1], torch.zeros_like(N_update[1]), N_update[1])

    
    H[0] = H_update[1]
    M[0] = M_update[1]
    N[0] = N_update[1]
    # Fx_list.append(dd(Fx))
    # Fy_list.append(dd(Fy))

    
    if itime % anim_interval == 0:
        # only part of the data is saved
        eta_list.append(dd(H_update[1]))
        u_list.append(dd(M_update[1]))
        v_list.append(dd(N_update[1]))
        
        # 从 M_update 与 N_update 计算 u, v 分量
        skip = params.Nx//20
        u_center = dd(u2rho(M_update[1],'expand'))
        v_center = dd(v2rho(N_update[1],'expand'))
    
        # 若 u_center 与 v_center 尺寸不一致，则取尺寸交集
        min_rows = min(u_center.shape[0], v_center.shape[0])
        min_cols = min(u_center.shape[1], v_center.shape[1])
        
        uc = u_center[:min_rows:skip, :min_cols:skip]
        vc = v_center[:min_rows:skip, :min_cols:skip]
        mag = np.sqrt(uc**2 + vc**2)
        Xc = X[:min_rows:skip, :min_cols:skip]
        Yc = Y[:min_rows:skip, :min_cols:skip]
        
        
        fig = plt.figure(figsize=(12,8))
        pcolormesh(X, Y, eta_list[-1], vmin=-1, vmax=1, cmap=plt.cm.RdBu_r);
        colorbar()
        axis('equal')
        xlim(120, 300)  # 经度范围
        ylim(-50, 60)   # 纬度范围
        from scipy.io import loadmat
        cc = loadmat(r'E:\OneDrive\plot\coastlines.mat')
        plot(cc['coastlon'],cc['coastlat'],'k-',linewidth = .1)
        quiver(Xc, Yc, uc, vc, scale=10000, color='k')     
        xlabel("Lon", fontname="serif", fontsize=12)
        ylabel("Lat", fontname="serif", fontsize=12)
        title("Stage at an instant in time: " + f"{itime*params.dt}" + " second")
        show()

max_eta = np.max(np.stack(eta_list),0)
figure(figsize=(12,8))
pcolormesh(X,Y,max_eta,vmin=0,vmax=1);colorbar()
axis('equal')
xlim(120, 300)  # 经度范围
ylim(-50, 60)   # 纬度范围
plot(cc['coastlon'],cc['coastlat'],'k-',linewidth=.01)
title('Max Tsunami Wave [m]')



v_array = np.array(v_list)
u_array = np.array(u_list)
eta = np.array(eta_list)
# Fx_array = np.array(Fx_list)
# Fy_array = np.array(Fy_list)
# sustr_array = np.array(sustr_list)
# svstr_array = np.array(svstr_list)
# np.save('Fx_large_array.npy', Fx_array)
# np.save('Fy_large_array.npy', Fy_array) 


# 计算 u_center 和 v_center（保持原逻辑）
u_center = np.empty((u_array.shape[0], x.shape[0], y.shape[0]))
u_center[:, 1:-1, :] = 0.5 * (u_array[:, :-1, :] + u_array[:, 1:, :])
u_center[:, 0, :] = u_array[:, 0, :]  # 边界处理
u_center[:, -1, :] = u_array[:, -1, :]

v_center = np.empty((v_array.shape[0], x.shape[0], y.shape[0]))
v_center[:, :, 1:-1] = 0.5 * (v_array[:, :, :-1] + v_array[:, :, 1:])
v_center[:, :, 0] = v_array[:, :, 0]  # 边界处理
v_center[:, :, -1] = v_array[:, :, -1]

# 创建统一的 Dataset
ds_combined = xr.Dataset(
    {
        'eta': (['time', 'lat', 'lon'], np.array(eta_list).swapaxes(1,2)),  # eta 在中心点
        'u': (['time', 'lat', 'lon' ], u_center.swapaxes(1,2)),      # u 插值到中心点
        'v': (['time', 'lat', 'lon' ], v_center.swapaxes(1,2))       # v 插值到中心点
    },
    coords={
        'lat': y,                             # 纬度中心点坐标
        'lon': x,                             # 经度中心点坐标
        'time': np.arange(len(eta_list))*500      # 时间坐标
    },
    attrs={
        'description': 'Combined output of eta, u, v, and their center-interpolated values',
    }
)

ds_combined['time'].encoding.update({
    'units': 'minutes since 2011-03-11 05:46:00',
    'calendar': 'proleptic_gregorian'
})

ds_combined.to_netcdf('his.tsunami_grid_0.1.nc')
# np.save('manning_large_distribution.npy', params.manning.numpy())    