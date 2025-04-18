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
from All_params import Params_small,Params_large,Params_tsunami                                            
from dynamics_train import mass_cartesian_torch,momentum_nonlinear_cartesian_torch
import torch.nn as nn
import torch.optim as optim
from CNN_manning_diff import CNN1
import torch
import time  # 在文件头部导入
import scipy.io as sio
#from radam import RAdam  # 或 pip install radam，具体看第三方库
from torchviz import make_dot      
import torch.utils.checkpoint as checkpoint


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

case = 'tsunami'

params = Params_tsunami(device)

dt = params.dt
NT = params.NT

ds_grid = xr.open_dataset('Data/tsunami_grid_1.nc')
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

manning_array = torch.zeros((2, params.Nx+1, params.Ny+1))

X_tensor = torch.stack([eta_tensor, u_resized, v_resized], dim=1)  
Y_tensor = torch.tensor(manning_array, dtype=torch.float64)        
X_tensor.requires_grad = True

# 从 mat 文件中加载数据，返回的是一个字典
data_filtered = sio.loadmat('Data/dart_data_filtered.mat')

# 提取出滤波后的数据，这里假设保存的键名为 'h_all_filtered'
dart_array = data_filtered['h_all_highpass']

# 转换为 torch tensor，数据类型为 float32
dart_tensor = torch.tensor(dart_array, dtype=torch.float64, device=device) 


# 文件中应存储 'lon'、'lat'（以及其它数据）
lon = data_filtered['lon']  # 可能形状为 (16, 1) 或 (16,)
lat = data_filtered['lat']

# 将 lon, lat 压缩成一维数组
lon = np.squeeze(lon)
lat = np.squeeze(lat)

# print("浮标经度：", lon)
# print("浮标纬度：", lat)

# 如果保存的经纬度就是模拟域坐标（例如 x 和 y 均在 [0, Lx] 和 [0, Ly] 范围内），
# 下面直接利用线性空间找到最近的网格索引。如果不是，需要根据实际情况进行坐标变换。

x_grid = np.linspace(120, 220, params.Nx + 1)
# 对于纬度，我们这里用 buoy 数据的最小值和最大值，也可以根据实际情况设定固定范围
y_grid = np.linspace(-20, 60, params.Ny + 1)

# 依次遍历每个浮标，找出最近的网格索引
meas_indices = []  # 用于存放 16 个点位在 H 中的索引，如 [(ix0, iy0), (ix1, iy1), ...]
for i in range(len(lon)):
    # 这里假设 lon 对应 x 方向，lat 对应 y 方向
    ix = np.argmin(np.abs(x_grid - lon[i]))
    iy = np.argmin(np.abs(y_grid - lat[i]))
    meas_indices.append((ix, iy))
    print(f"浮标 {i}（经度 {lon[i]}， 纬度 {lat[i]}） 对应网格索引: (ix={ix}, iy={iy})")

meas_indices = np.array(meas_indices)

input_channel = 3
model = CNN1(shapeX=input_channel).to(device)
model = model.double().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

torch.autograd.set_detect_anomaly(True)

# 优化后的训练循环：直接利用整个训练序列（X_tensor）进行多步模拟

eta_list = []
u_list = []
v_list = []
train_loss_history = []
manning_list = []
Fx_list = []
Fy_list = []
loss_list = []

#manning = torch.rand((params.Nx+1, params.Ny+1)).requires_grad_(True).to(device)
last_epoch_H_list = []
last_epoch_M_list = []
last_epoch_N_list = []
last_epoch_manning_list = []

# 假定 land_mask 已经是 (Nx+1, Ny+1) 的布尔张量

# 先提取出常数驱动
Wx0 = Wx[0]        # shape (Nx+1, Ny+1)，全 0
Wy0 = Wy[0]        # shape (Nx+1, Ny+1)，全 0
Pa0 = Pa[0]        # shape (Nx+1, Ny+1)，全 1000

# ========================
# 定义 simulation_step（保持不变）
# ========================

def simulation_step(H, M, N, params, manning):
    H_new = mass_cartesian_torch(H, Z, M, N, params)
    
    H0, H1 = H_new[0], H_new[1]
    H1 = torch.where(land_mask, torch.zeros_like(H1), H1)
    H_upd = torch.stack([H0, H1], dim=0)
    
    M_new, N_new, _, _ = momentum_nonlinear_cartesian_torch(
        H_upd, Z, M, N, Wx0, Wy0, Pa0, params, manning
    )

    M0, M1 = M_new[0], M_new[1]
    M1 = torch.where(land_mask[:-1, :], torch.zeros_like(M1), M1)
    M_upd = torch.stack([M0, M1], dim=0)

    N0, N1 = N_new[0], N_new[1]
    N1 = torch.where(land_mask[:, :-1], torch.zeros_like(N1), N1)
    N_upd = torch.stack([N0, N1], dim=0)

    return H_upd, M_upd, N_upd

manning0 = manning_array[1].to(device)

num_epochs = 100
inner_steps = 20

for epoch in range(1, num_epochs+1):
    model.train()
    # 初始化 H、M、N
    H = torch.zeros((2, params.Nx+1, params.Ny+1), device=device, dtype=torch.float64); H[0] = H_ini
    M = torch.zeros((2, params.Nx,   params.Ny+1), device=device, dtype=torch.float64)
    N = torch.zeros((2, params.Nx+1, params.Ny),   device=device, dtype=torch.float64)
    t = 0
    training_enabled = False
    total_epoch_loss = 0.0

    while t < params.NT:
        # —— plain step —— 
        H_pred, M_pred, N_pred = simulation_step(H, M, N, params, manning0)
        # detach + roll
        H = torch.roll(H_pred.detach(), shifts=-1, dims=0)
        M = torch.roll(M_pred.detach(), shifts=-1, dims=0)
        N = torch.roll(N_pred.detach(), shifts=-1, dims=0)
        t += 1

        mask = torch.zeros_like(H_pred[1], dtype=torch.bool, device=device)
        mask[meas_indices[:, 0], meas_indices[:, 1]] = True
        simulated = torch.masked_select(H_pred[1], mask)
        num_zeros = int((simulated == 0).sum().item())
        print(f"  After step {t}/{params.NT}: zeros in simulated = {num_zeros}/{simulated.numel()}")

        if not training_enabled:
            if num_zeros > 12:
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
                manning_pred = model(inp).squeeze(0).squeeze(0)
        
                # ④ 在相同基准上做一次仿真：mass→momentum
                H_mid, M_mid, N_mid = simulation_step(H_seg, M_seg, N_seg, params, manning_pred)
        
                H_prev = H_mid[1]
                M_prev = M_mid[1]
                N_prev = N_mid[1]
                # 构造新的 2‑step 数组，index 0 用刚更新的状态
                H_stack = torch.stack([H_prev, H_prev], dim=0)
                M_stack = torch.stack([M_prev, M_prev], dim=0)
                N_stack = torch.stack([N_prev, N_prev], dim=0)
                H_extra = mass_cartesian_torch(H_stack, Z, M_stack, N_stack, params)
                
                H0e, H1e = H_extra[0], H_extra[1]
                H1e = torch.where(land_mask, torch.zeros_like(H1e), H1e)
                H_extra = torch.stack([H0e, H1e], dim=0)
        
                # ⑥ 计算 loss（同一个 time-step, 一直用 H_extra[1]）
                simulated_seg = torch.masked_select(H_extra[1], mask)
                loss = criterion(simulated_seg, dart_tensor[:, t+1])
                print(f"    inner {inner+1}/{inner_steps} — loss = {loss.item():.16f}")
        
                # ⑦ 反向并更新模型；但不要修改 H_base
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_epoch_loss += loss.item()
        
            # —— inner 完毕后，用那次 H_mid 结果推进主循环状态 —— 
            # detach 并滚动一格，让下一个 t+1 用这个最新状态
            H = torch.roll(H_mid.detach(), shifts=-1, dims=0)
            M = torch.roll(M_mid.detach(), shifts=-1, dims=0)
            N = torch.roll(N_mid.detach(), shifts=-1, dims=0)
            t +=1
    print(f"Epoch {epoch} loss={total_epoch_loss:.6f}")



# # —— 训练主循环 （方法二 + 打印） ——  
# K = 2
# inner_steps = 3
# num_epochs = 100

# for epoch in range(1, num_epochs + 1):
#     model.train()
#     optimizer.zero_grad()

#     H = torch.zeros((2, params.Nx+1, params.Ny+1), device=device, dtype=torch.float64)
#     H[0] = H_ini
#     M = torch.zeros((2, params.Nx,   params.Ny+1), device=device, dtype=torch.float64)
#     N = torch.zeros((2, params.Nx+1, params.Ny),   device=device, dtype=torch.float64)

#     t = 0
#     total_epoch_loss = 0.0
#     training_enabled = False

#     while t < params.NT:
#         L = min(K, params.NT - t)
#         print(f"\n[Epoch {epoch}] Segment start t={t}")

#         # —— 普通步：先用零场或默认 manning0 推进一次
#         manning0 = torch.zeros_like(H[0])
#         H_pred, M_pred, N_pred = simulation_step(H, M, N, params, manning0)

#         # 计算观测点 simulated
#         mask = torch.zeros_like(H_pred[1], dtype=torch.bool, device=device)
#         mask[meas_indices[:, 0], meas_indices[:, 1]] = True
#         simulated = torch.masked_select(H_pred[1], mask)
#         num_zeros = int((simulated == 0).sum().item())
#         print(f"  After plain step: zeros in simulated = {num_zeros}/{simulated.numel()}")

#         # 如果还没全非零，就只推进状态，不训练
#         if not training_enabled and num_zeros > 0:
#             H[0] = H_pred[1].detach()
#             M[0]= M_pred[1].detach()
#             N[0] = N_pred[1].detach()
#             t += 1
#             continue

#         # 开始训练
#         if not training_enabled:
#             print(f"  → All observed points non-zero. Enabling training from t={t}.")
#             H[0] = H_pred[1].detach()
#             M[0]= M_pred[1].detach()
#             N[0] = N_pred[1].detach()
#             training_enabled = True

#             inp = torch.stack([
#                 H[0].to(torch.float32),
#                 u2rho(M[0], mode='expand').to(torch.float32),
#                 v2rho(N[0], mode='expand').to(torch.float32),
#             ], dim=0).unsqueeze(0)
            
#             for _ in range(L):
#                 manning_pred = model(inp).squeeze(0).squeeze(0).to(torch.float64)
#                 H_seg, M_seg, N_seg = simulation_step(
#                     H, M, N, params, manning_pred
#                 )
#                 inp = torch.stack([
#                     H_seg[0].to(torch.float32),
#                     u2rho(M_seg[0], mode='expand').to(torch.float32),
#                     v2rho(N_seg[0], mode='expand').to(torch.float32),
#                 ], dim=0).unsqueeze(0)

#                 H[0] = H_seg[1]
#                 M[0] = M_seg[1]
#                 N[0] = N_seg[1]

#             simulated_seg = torch.masked_select(H_seg[1], mask)
#             seg_loss = criterion(simulated_seg, dart_tensor[:, t+L-1])
#             print(f" seg_loss = {seg_loss.item():.12f}")

#             optimizer.zero_grad()
#             seg_loss.backward()
#             optimizer.step()

#         total_epoch_loss += seg_loss.item()
#         H.detach()
#         M.deatch()
#         N.detach()
#         t += L

#     print(f"Epoch {epoch}/{num_epochs}  total loss = {total_epoch_loss:.6f}")



# def simulation_step(H, M, N, params, manning):
#     # 1) 质量方程
#     H_new = mass_cartesian_torch(H, Z, M, N, params)
#     # 2) 动量方程（统一用 Wx0,Wy0,Pa0）
#     M_new, N_new, _, _ = momentum_nonlinear_cartesian_torch(
#         H_new, Z, M, N, Wx0, Wy0, Pa0, params, manning
#     )
#     # 3) 掩模（都用新的 tensor，不 in‑place）
#     H0, H1 = H_new[0], H_new[1]
#     H1 = torch.where(land_mask, torch.zeros_like(H1), H1)
#     H_upd = torch.stack([H0, H1], dim=0)

#     M0, M1 = M_new[0], M_new[1]
#     M1 = torch.where(land_mask[:-1,:], torch.zeros_like(M1), M1)
#     M_upd = torch.stack([M0, M1], dim=0)

#     N0, N1 = N_new[0], N_new[1]
#     N1 = torch.where(land_mask[:,:-1], torch.zeros_like(N1), N1)
#     N_upd = torch.stack([N0, N1], dim=0)

#     return H_upd, M_upd, N_upd


# # —— 训练主循环 —— 
# K = 2
# num_epochs = 100

# for epoch in range(1, num_epochs+1):
#     model.train()
#     optimizer.zero_grad()

#     # 1) 每个 epoch 从零开始，但不 detach（要把各段串起来）
#     H = torch.zeros((2, params.Nx+1, params.Ny+1), device=device, dtype=torch.float64)
#     H[0] = H_ini
#     M = torch.zeros((2, params.Nx,   params.Ny+1), device=device, dtype=torch.float64)
#     N = torch.zeros((2, params.Nx+1, params.Ny),   device=device, dtype=torch.float64)

#     t = 0
#     total_epoch_loss = 0.0

#     # 2) 分段推进，但段与段之间不 detach
#     while t < params.NT:
#         L = min(K, params.NT - t)

#         # 构造本段第 0 步的模型输入
#         inp = torch.stack([
#             H[0].to(torch.float32),
#             u2rho(M[0], mode='expand').to(torch.float32),
#             v2rho(N[0], mode='expand').to(torch.float32),
#         ], dim=0).unsqueeze(0)  # (1,3,H,W)

#         H_pred, M_pred, N_pred = H, M, N

#         # 段内连续 L 步前向
#         for _ in range(L):
#             manning = model(inp).squeeze(0).squeeze(0).to(torch.float64)
#             H_pred, M_pred, N_pred = simulation_step(
#                 H_pred, M_pred, N_pred, params, manning
#             )
#             inp = torch.stack([
#                 H_pred[0].to(torch.float32),
#                 u2rho(M_pred[0], mode='expand').to(torch.float32),
#                 v2rho(N_pred[0], mode='expand').to(torch.float32),
#             ], dim=0).unsqueeze(0)

#         # 段末计算一次 loss
#         mask = torch.zeros_like(H_pred[1], dtype=torch.bool, device=device)
#         mask[meas_indices[:,0], meas_indices[:,1]] = True
#         simulated = torch.masked_select(H_pred[1], mask)
#         measured  = dart_tensor[:, t+L-1]  # 已经是 tensor 且在 device 上
#         seg_loss  = criterion(simulated, measured)
#         print(f"loss = {seg_loss:.12f}")
#         # 立即反向并更新这一段的梯度
#         seg_loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         total_epoch_loss += seg_loss.item()

#         # **注意：这里不 detach**，下段直接接着用同一个图
#         H = H_pred.detach()
#         M = M_pred.detach()
#         N = N_pred.detach()
#         t += L

#     print(f"Epoch {epoch}/{num_epochs}  total loss = {total_epoch_loss:.6f}")






# 保存最后一个 epoch 的每个时间步状态和 loss
np.save('eta_list_train_manning.npy', np.array(last_epoch_H_list))
np.save('u_list_train_manning.npy', np.array(last_epoch_M_list))
np.save('v_list_train_manning.npy', np.array(last_epoch_N_list))
last_epoch_manning_list
np.save('manning_list_train_manning.npy', np.array(last_epoch_manning_list))
# 保存整个 epoch 平均 loss 历史
np.save('train_loss_history_manning.npy', np.array(train_loss_history))
torch.save(model.state_dict(), 'model_checkpoint_manning.pth')
print("Training complete and data saved.")
    