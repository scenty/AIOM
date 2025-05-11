import numpy as np
#import vis_tools
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from J65_torch import *
from matplotlib.pyplot import *
import xarray as xr
import torch
import torch.nn.functional as F
from tool_train import ddx,ddy,rho2u,rho2v,v2rho,u2rho,dd
from tool_train import ududx_up,vdudy_up,udvdx_up,vdvdy_up
from All_params import Params_large,Params_tsunami                                            
from dynamics_H import mass_cartesian_torch,momentum_nonlinear_cartesian_torch
import torch.nn as nn
import torch.optim as optim
from CNN_manning_diff import CNN1
import torch
import time  # 在文件头部导入
import scipy.io as sio
#from radam import RAdam  # 或 pip install radam，具体看第三方库
# from torchviz import make_dot      
import torch.utils.checkpoint as checkpoint
from visualization import plot_dart


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

case = 'tsunami'

params = Params_tsunami(device)

dt = params.dt
NT = params.NT

ds_grid = xr.open_dataset('Data/tsunami_grid_0.1.nc')
x = torch.from_numpy(ds_grid['lon'][:].values)
y = torch.from_numpy(ds_grid['lat'][:].values)

X, Y = torch.meshgrid(x, y, indexing='ij')
X = X.to(device)
Y = Y.to(device)

Z = torch.from_numpy(ds_grid['z'].values.astype(np.float64)).to(device)
Z = Z.T
Z = -torch.nan_to_num(Z, nan=0.0)
Z[Z<params.dry_limit] = params.dry_limit
land_mask = torch.from_numpy(ds_grid['mask'].values).T
land_mask = land_mask.to(torch.bool).to(device)
# 初始场
H_ini = torch.from_numpy(ds_grid['zeta'].values.astype(np.float64))
H_ini = H_ini.T  # 调整为 [51, 81]，与模型网格一致
H_ini = torch.nan_to_num(H_ini, nan=0.0)
H = torch.zeros((2, params.Nx+1, params.Ny+1), dtype=torch.float64).to(device)
H[0] = H_ini  

M = torch.zeros((2, params.Nx, params.Ny+1)).to(device)
N = torch.zeros((2, params.Nx+1, params.Ny)).to(device)


Wx = np.ones((1,X.shape[0],X.shape[1]))*0 + 0
Wy = np.ones((1,X.shape[0],X.shape[1]))*0 
Wx = torch.from_numpy(Wx).to(device)
Wy = torch.from_numpy(Wy).to(device)
Pa = Wx * 0 + 1000
    
    
eta_tensor = torch.tensor(H, dtype=torch.float64).to(device)
u_tensor = torch.tensor(M, dtype=torch.float64).to(device)
v_tensor = torch.tensor(N, dtype=torch.float64).to(device)

target_size = eta_tensor.shape[1:]  

u_resized = F.interpolate(u_tensor.unsqueeze(1), 
                          size=target_size, mode='bilinear', align_corners=False).squeeze(1)
v_resized = F.interpolate(v_tensor.unsqueeze(1), 
                          size=target_size, mode='bilinear', align_corners=False).squeeze(1)

Z_pred = torch.zeros((params.Nx+1, params.Ny+1)).to(device)

X_tensor = torch.stack([eta_tensor, u_resized, v_resized], dim=1)  
Y_tensor = torch.tensor(Z_pred, dtype=torch.float64)        
X_tensor.requires_grad = True

# 1. 读 mat
data_filtered = sio.loadmat('Data/dart_data_filtered.mat')
dart_array = data_filtered['h_all_highpass']
dart_tensor = torch.tensor(dart_array, dtype=torch.float64, device=device) 
dart_time = data_filtered['time'].squeeze(0)
dart_time = np.array([np.datetime64(x[0]) for x in dart_time]) 
dart_list = data_filtered['buoy_ids']
# 文件中应存储 'lon'、'lat'（以及其它数据）
lon = data_filtered['lon'].squeeze()  # 可能形状为 (16, 1) 或 (16,)
lat = data_filtered['lat'].squeeze()
# 去除21416
dart_list = np.delete(dart_list,6)
dart_tensor = torch.cat((dart_tensor[:6],dart_tensor[7:]))
lon = np.delete(lon,6)
lat = np.delete(lat,6)

# 4. 转 torch（只需按需做一次）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_grid = np.linspace(120, 220, params.Nx + 1)
# 对于纬度，我们这里用 buoy 数据的最小值和最大值，也可以根据实际情况设定固定范围
y_grid = np.linspace(-20, 60, params.Ny + 1)

# 依次遍历每个浮标，找出最近的网格索引
meas_indices = []

for i in range(len(lon)):
    # 这里假设 lon 对应 x 方向，lat 对应 y 方向
    ix = np.argmin(np.abs(params.lon - lon[i]))
    iy = np.argmin(np.abs(params.lat - lat[i]))
    meas_indices.append((ix, iy))
    print(f"浮标 {dart_list[i]}（经度 {lon[i]}， 纬度 {lat[i]}） 对应网格索引: (ix={ix}, iy={iy})")

plot_dart(dart_time, dd(dart_tensor), dart_list)
#%%
meas_indices = np.array(meas_indices)

input_channel = 3
model = CNN1(shapeX=input_channel).to(device)
model = model.double().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

torch.autograd.set_detect_anomaly(True)

# 优化后的训练循环：直接利用整个训练序列（X_tensor）进行多步模拟

eta_list = []
u_list = []
v_list = []
train_loss_history = []
Z_recorrected_list = []
Fx_list = []
Fy_list = []
loss_list = []

#manning = torch.rand((params.Nx+1, params.Ny+1)).requires_grad_(True).to(device)
last_epoch_H_list = []
last_epoch_M_list = []
last_epoch_N_list = []
last_epoch_recorrected_list = []

# 假定 land_mask 已经是 (Nx+1, Ny+1) 的布尔张量

# 先提取出常数驱动
Wx0 = Wx[0]        # shape (Nx+1, Ny+1)，全 0
Wy0 = Wy[0]        # shape (Nx+1, Ny+1)，全 0
Pa0 = Pa[0]        # shape (Nx+1, Ny+1)，全 1000

# ========================
# 定义 simulation_step（保持不变）
# ========================

def simulation_step(H, M, N, params, Z_recorrected):

    Z_eff = Z + Z_recorrected
    H_new = mass_cartesian_torch(H, Z_eff, M, N, params)
    
    H0, H1 = H_new[0], H_new[1]
    H0 = torch.where(land_mask, torch.zeros_like(H0), H0)
    H1 = torch.where(land_mask, torch.zeros_like(H1), H1)
    H_upd = torch.stack([H0, H1], dim=0)
    
    M_new, N_new, _, _ = momentum_nonlinear_cartesian_torch(
        H_upd, Z_eff, M, N, Wx0, Wy0, Pa0, params, Z_recorrected
    )

    M0, M1 = M_new[0], M_new[1]
    M0 = torch.where(land_mask[:-1, :], torch.zeros_like(M0), M0)
    M1 = torch.where(land_mask[:-1, :], torch.zeros_like(M1), M1)
    
    M_upd = torch.stack([M0, M1], dim=0)

    N0, N1 = N_new[0], N_new[1]
    N0 = torch.where(land_mask[:, :-1], torch.zeros_like(N0), N0)
    N1 = torch.where(land_mask[:, :-1], torch.zeros_like(N1), N1)
    N_upd = torch.stack([N0, N1], dim=0)

    return H_upd, M_upd, N_upd

num_epochs = 10
inner_steps = 5

model_dart_list = []
last_epoch_H_list      = []
last_epoch_M_list      = []
last_epoch_N_list      = []
last_epoch_Z_pred_list = []

for epoch in range(1, num_epochs+1):
    model.train()
    # 初始化 H、M、N
    
    H = torch.zeros((2, params.Nx+1, params.Ny+1), device=device, dtype=torch.float64); H[0] = H_ini
    M = torch.zeros((2, params.Nx,   params.Ny+1), device=device, dtype=torch.float64)
    N = torch.zeros((2, params.Nx+1, params.Ny),   device=device, dtype=torch.float64)
    t = 0
    training_enabled = False
    total_epoch_loss = 0.0
    record = (epoch == num_epochs)

    while t < params.NT:
        # —— plain step —— 
        H_pred, M_pred, N_pred = simulation_step(H, M, N, params, Z_pred)
        # detach + roll
        H = torch.roll(H_pred.detach(), shifts=-1, dims=0)
        M = torch.roll(M_pred.detach(), shifts=-1, dims=0)
        N = torch.roll(N_pred.detach(), shifts=-1, dims=0)
        
        if record:
            last_epoch_H_list.append(H_pred[1].detach().cpu().numpy())
            last_epoch_M_list.append(M_pred[1].detach().cpu().numpy())
            last_epoch_N_list.append(N_pred[1].detach().cpu().numpy())
            last_epoch_Z_pred_list.append(Z_pred.detach().cpu().numpy())
        
        t += 1
        # 阈值（可根据需要调整）
        threshold = 1e-4
        
        # 在原代码里替换：
        mask = torch.zeros_like(H_pred[1], dtype=torch.bool, device=device)
        simulated = H_pred[1][meas_indices[:,0], meas_indices[:,1]]
        dart_val = dd(simulated)
        model_dart_list.append(dart_val)
        
        # 统计“近零”的数量，而不是严格等于零
        num_near_zero = int((simulated.abs() < threshold).sum().item())
        print(f"  After step {t}/{params.NT}: near-zero (<{threshold}) count = {num_near_zero}/{simulated.numel()}")
    # —— 计算 dd(simulated) 并存储 —— 
        
        if t%20 == 0:
            # 面场可视化
            fig = plt.figure(figsize=(12,8))
            #var = dd(u2rho(M[-1],'expand'))
            var = dd(H_pred[-1])
            pcolormesh(dd(X), dd(Y), var, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r);
            colorbar()
            #axis('equal')
            xlim(120, 300)  # 经度范围
            ylim(-50, 60)   # 纬度范围
            from scipy.io import loadmat
            cc = loadmat(r'E:\OneDrive\plot\coastlines.mat')
            plot(cc['coastlon'],cc['coastlat'],'k-',linewidth = .1)
            #quiver(Xc, Yc, uc, vc, scale=10000, color='k')     
            plot(lon,lat,'k.')
            xlabel("Lon", fontname="serif", fontsize=12)
            ylabel("Lat", fontname="serif", fontsize=12)
            title("Stage at an instant in time: " + f"{t*params.dt}" + " second")
            show()
            
            # 站点可视化
            
            var = np.array(model_dart_list)
            plot_dart(dart_time, dd(dart_tensor), dart_list, t, var)
    
        if not training_enabled:
            if num_near_zero > 8:
                continue
            else:
                print("→ 开启训练")
                training_enabled = True
        # —— 训练分支 —— 
        if training_enabled:
            # ① 保存本 time-step 的基准状态
            H_base, M_base, N_base = H.clone(), M.clone(), N.clone()
        
            H_mid = M_mid = N_mid = None
            for inner in range(inner_steps):
                # ② 每次都从基准状态 clone
                H_seg, M_seg, N_seg = H_base.clone(), M_base.clone(), N_base.clone()
        
                # ③ 构造模型输入
                inp = torch.stack([
                    H_seg[0].to(torch.float32),
                    u2rho(M_seg[0], mode='expand').to(torch.float64),
                    v2rho(N_seg[0], mode='expand').to(torch.float64),
                ], dim=0).unsqueeze(0)
                Z_pred = model(inp).squeeze(0).squeeze(0)
                Z_pred = torch.where(land_mask, torch.zeros_like(Z_pred), Z_pred)

                # ④ 在相同基准上做一次仿真：mass→momentum
                H_mid, M_mid, N_mid = simulation_step(H_seg, M_seg, N_seg, params, Z_pred)
        
                H_prev = H_mid[1]
                M_prev = M_mid[1]
                N_prev = N_mid[1]
                # 构造新的 2‑step 数组，index 0 用刚更新的状态
                H_stack = torch.stack([H_prev, H_prev], dim=0)
                M_stack = torch.stack([M_prev, M_prev], dim=0)
                N_stack = torch.stack([N_prev, N_prev], dim=0)
                H_extra = mass_cartesian_torch(H_stack, Z, M_stack, N_stack, params)
                
                H0e, H1e = H_extra[0], H_extra[1]
                # H0e = torch.where(land_mask, torch.zeros_like(H1e), H0e)
                # H1e = torch.where(land_mask, torch.zeros_like(H1e), H1e)
                H_extra = torch.stack([H0e, H1e], dim=0)
        
                # ⑥ 计算 loss（同一个 time-step, 一直用 H_extra[1]）
                simulated_seg = H_extra[1][meas_indices[:,0], meas_indices[:,1]]
                dart_val = dd(simulated_seg)
                model_dart_list.append(dart_val)
                loss = criterion(simulated_seg, dart_tensor[:, t+2])
                print(f"    inner {inner+1}/{inner_steps} — loss = {loss.item():.16f}")
        
                # ⑦ 反向并更新模型；但不要修改 H_base
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_epoch_loss += loss.item()
        
            # —— inner 完毕后，用那次 H_mid 结果推进主循环状态 —— 
            if record:
                last_epoch_H_list.append(   H_mid[1].detach().cpu().numpy() )
                last_epoch_M_list.append(   M_mid[1].detach().cpu().numpy() )
                last_epoch_N_list.append(   N_mid[1].detach().cpu().numpy() )
                last_epoch_Z_pred_list.append( Z_pred.detach().cpu().numpy() )

            H = torch.roll(H_mid.detach(), shifts=-1, dims=0)
            M = torch.roll(M_mid.detach(), shifts=-1, dims=0)
            N = torch.roll(N_mid.detach(), shifts=-1, dims=0)
            t += 1
    
    train_loss_history.append(total_epoch_loss / params.NT)        
    print(f"Epoch {epoch} loss={total_epoch_loss:.6f}")


# 保存最后一个 epoch 的每个时间步状态和 loss
np.save(r'D:/eta_list_train_H.npy', np.array(last_epoch_H_list))
np.save(r'D:/u_list_train_H.npy', np.array(last_epoch_M_list))
np.save(r'D:/v_list_train_H.npy', np.array(last_epoch_N_list))
np.save(r'D:/Z_pred_list_train_H.npy', np.array(last_epoch_Z_pred_list))
# 保存整个 epoch 平均 loss 历史
np.save(r'D:/train_loss_history_H.npy', np.array(train_loss_history))
torch.save(model.state_dict(), r'D:/model_checkpoint_H.pth')
print("Training complete and data saved.")
    

