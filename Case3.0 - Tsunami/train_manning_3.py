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
# from torchviz import make_dot      

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

case = 'tsunami'

params = Params_tsunami()

dt = params.dt
NT = params.NT

ds_grid = xr.open_dataset('Data/tsunami_grid_1.nc')
x = torch.from_numpy(ds_grid['lon'][:].values)
y = torch.from_numpy(ds_grid['lat'][:].values)

X, Y = torch.meshgrid(x, y, indexing='ij')
X = X.to(device)
Y = Y.to(device)

Z = torch.from_numpy(ds_grid['z'].values.astype(np.float64)).to(device)
Z = Z.T  # 调整为 [51, 81]
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
Wx = torch.from_numpy(Wx)
Wy = torch.from_numpy(Wy)
Pa = Wx * 0 + 1000
    
    
eta_tensor = torch.tensor(H, dtype=torch.float32).to(device)
u_tensor = torch.tensor(M, dtype=torch.float32).to(device)
v_tensor = torch.tensor(N, dtype=torch.float32).to(device)

target_size = eta_tensor.shape[1:]  

u_resized = F.interpolate(u_tensor.unsqueeze(1), 
                          size=target_size, mode='bilinear', align_corners=False).squeeze(1)
v_resized = F.interpolate(v_tensor.unsqueeze(1), 
                          size=target_size, mode='bilinear', align_corners=False).squeeze(1)

manning_array = torch.zeros((2, params.Nx+1, params.Ny+1))

X_tensor = torch.stack([eta_tensor, u_resized, v_resized], dim=1)  
Y_tensor = torch.tensor(manning_array, dtype=torch.float32)        


# 从 mat 文件中加载数据，返回的是一个字典
data_filtered = sio.loadmat('Data/dart_data_filtered.mat')

# 提取出滤波后的数据，这里假设保存的键名为 'h_all_filtered'
dart_array = data_filtered['h_all_filtered']

# 转换为 torch tensor，数据类型为 float32
dart_tensor = torch.tensor(dart_array, dtype=torch.float32)


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

x_grid = np.linspace(120, 300, params.Nx + 1)
# 对于纬度，我们这里用 buoy 数据的最小值和最大值，也可以根据实际情况设定固定范围
y_grid = np.linspace(-60, 60, params.Ny + 1)

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
model.float()  
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

torch.autograd.set_detect_anomaly(True)

# 优化后的训练循环：直接利用整个训练序列（X_tensor）进行多步模拟
num_epochs = 100
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

for epoch in range(num_epochs):
    model.train()
    
    epoch_loss_sum = 0.0  # 累加当前 epoch 内所有时间步的 loss
    current_input = X_tensor[0].unsqueeze(0).to(device).float()
    
    # 重新初始化状态，每个 epoch 内从零开始
    M = torch.zeros((2, params.Nx, params.Ny+1)).to(device)
    N = torch.zeros((2, params.Nx+1, params.Ny)).to(device)
    H = torch.zeros((2, params.Nx+1, params.Ny+1)).to(device)
    H[0] = H_ini  # 现在 H_ini 中不含 NaN
    
    for t in range(params.NT):
        # 保存当前时间步的初始状态
        H_init = H.clone()
        M_init = M.clone()
        N_init = N.clone()
        inner_steps = 3
    
        for i in range(inner_steps):
            manning = model(current_input).squeeze(0).squeeze(0)
            H_update = mass_cartesian_torch(H_init, Z, M_init, N_init, params)
            # 应用陆域掩模：陆地（mask==1）处水位与动量置零
            H_update[1] = torch.where(land_mask, torch.zeros_like(H_update[1]), H_update[1])
            
            M_update, N_update, _ , _ = momentum_nonlinear_cartesian_torch(H_update, Z, M, N, Wx[0], Wy[0], Pa[0], params, manning)
            
            # 对 M（尺寸：(Nx, Ny+1)）使用 land_mask 的前 Nx 行：
            M_update[1] = torch.where(land_mask[:-1, :], torch.zeros_like(M_update[1]), M_update[1])
            
            # 对 N（尺寸：(Nx+1, Ny)）使用 land_mask 的前 Ny 列：
            N_update[1] = torch.where(land_mask[:, :-1], torch.zeros_like(N_update[1]), N_update[1])
            
            scale = 1
            # 如果当前仿真步和观测时刻对齐（300 s对应12个仿真步）
            if t % 12 == 0 and (t // 12) < dart_array.shape[1]:
                measured_idx = t // 12
                # 提取当前观测时刻16个点的测量值（注意 dart_array shape 为 [16, 145]）
                measured_values = torch.tensor(dart_array[:, measured_idx],
                                               device=device, dtype=torch.float32)
                # 从 H_update 中采样对应位置的水位值（假设 H_update[1] 的 shape 为 [Nx+1, Ny+1]）
                # meas_indices[:,0] -> x方向索引, meas_indices[:,1] -> y方向索引
                simulated_meas = H_update[1][meas_indices[:,0], meas_indices[:,1]]
                loss_eta = criterion(simulated_meas * scale, measured_values * scale)
            else:
                loss_eta = 0 * torch.mean(H_update[1])

            loss = loss_eta 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # 用最新的模型参数计算一次更新（不进行梯度更新）
        H_update = mass_cartesian_torch(H_init, Z, M_init, N_init, params)
        # 应用陆域掩模：陆地（mask==1）处水位与动量置零
        H_update[1] = torch.where(land_mask, torch.zeros_like(H_update[1]), H_update[1])
        
        M_update, N_update, _ , _ = momentum_nonlinear_cartesian_torch(H_update, Z, M, N, Wx[0], Wy[0], Pa[0], params, manning)
        
        # 对 M（尺寸：(Nx, Ny+1)）使用 land_mask 的前 Nx 行：
        M_update[1] = torch.where(land_mask[:-1, :], torch.zeros_like(M_update[1]), M_update[1])
        
        # 对 N（尺寸：(Nx+1, Ny)）使用 land_mask 的前 Ny 列：
        N_update[1] = torch.where(land_mask[:, :-1], torch.zeros_like(N_update[1]), N_update[1])
        
        # 更新状态，用于下一个时间步
        H[0] = H_update[1].detach()
        M[0] = M_update[1].detach()
        N[0] = N_update[1].detach()
        
        # 对 M_update、N_update 做插值（保证与 H_update 一致）
        target_size = H_update[1].shape
        M_resized = F.interpolate(M_update[1].unsqueeze(0).unsqueeze(0),
                                  size=target_size, mode='bilinear', align_corners=False) \
                                  .squeeze(0).squeeze(0)
        N_resized = F.interpolate(N_update[1].unsqueeze(0).unsqueeze(0),
                                  size=target_size, mode='bilinear', align_corners=False) \
                                  .squeeze(0).squeeze(0)
        current_state = torch.stack([H[0], M_resized, N_resized], dim=0)
        current_input = current_state.unsqueeze(0).detach()
        
        # 计算当前时间步最终 loss（同样只在观测时刻计算）
        if t % 12 == 0 and (t // 12) < dart_array.shape[1]:
            measured_idx = t // 12
            measured_values = torch.tensor(dart_array[:, measured_idx],
                                           device=device, dtype=torch.float32)
            simulated_meas = H_update[1][meas_indices[:,0], meas_indices[:,1]]
            final_loss_eta = criterion(simulated_meas * scale, measured_values * scale)
        else:
            final_loss_eta = 0 * torch.mean(H_update[1])
        
        epoch_loss_sum += final_loss_eta.item()
        
        if epoch == num_epochs - 1:
            last_epoch_H_list.append(dd(H_update[1]))
            last_epoch_M_list.append(dd(M_update[1]))
            last_epoch_N_list.append(dd(N_update[1]))
            last_epoch_manning_list.append(dd(manning))

    
    # 计算当前 epoch 内所有时间步的平均 loss，并保存到总的 loss 历史中
    epoch_avg_loss = epoch_loss_sum / params.NT
    train_loss_history.append(epoch_avg_loss.cpu().item())

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_avg_loss:.15f}")

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
    