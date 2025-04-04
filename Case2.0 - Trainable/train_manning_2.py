import numpy as np
#import vis_tools
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from J65_torch import *
from matplotlib.pyplot import *
import xarray as xr
import torch
import torch.nn.functional as F
from tool_train import ddx,ddy,rho2u,rho2v,v2rho,u2rho,dd
from tool_train import ududx_up,vdudy_up,udvdx_up,vdvdy_up
from All_params import Params_small,Params_large,Params_tsunami                                            
from dynamics import mass_cartesian_torch,momentum_nonlinear_cartesian_torch
import torch.nn as nn
import torch.optim as optim
from CNN_manning_diff import CNN1
import torch
import time  # 在文件头部导入
#from radam import RAdam  # 或 pip install radam，具体看第三方库
# from torchviz import make_dot      


def plot_stage_Point(var2plot, x_ind):
    # Parameters
    time2plot = len(var2plot)
    time = np.arange(time2plot) * dt

    # x_ind = 115
    y_ind = 20
    var2plot = [_[x_ind, y_ind] for _ in var2plot]
    fig, ax = plt.subplots(1, 1)
    plt.plot(time, var2plot, 'b.', label='numerical')
    plt.title("Stage at the centre of the  basin over time")
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Stage')
    fig.savefig("Stage_Rad" + f"{x_ind}" + ".jpg", format='jpg', dpi=300, bbox_inches='tight')
    plt.close()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

case = 'small'
params = Params_small(device)


# 读取 eta 数据
ds_eta = xr.open_dataset(f'out_{case}_eta.nc')
eta_array = ds_eta['eta'].values  # 或者直接使用 ds_eta.eta

# 读取 u 数据
ds_u = xr.open_dataset(f'out_{case}_u.nc')
u_array = ds_u['u'].values  # 或者直接使用 ds_u.u

# 读取 v 数据
ds_v = xr.open_dataset(f'out_{case}_v.nc')
v_array = ds_v['v'].values  # 或者直接使用 ds_v.v
 
dt = params.dt
NT = params.NT

x = torch.linspace(0, params.Lx, params.Nx + 1)
y = torch.linspace(0, params.Ly, params.Ny + 1)
X, Y = torch.meshgrid(x, y)
X = X.to(device)
Y = Y.to(device)
manning_distribution = np.load('manning_small_distribution.npy')
manning_array = manning_distribution[np.newaxis,:,:].repeat(NT.cpu(), axis=0)

# initial condition
M = torch.zeros((2, params.Nx, params.Ny+1)).to(device)
N = torch.zeros((2, params.Nx+1, params.Ny)).to(device)
H = torch.zeros((2, params.Nx+1, params.Ny+1)).to(device)
# 确保将 params.depth 转移到正确的设备上
Z = torch.ones((params.Nx + 1, params.Ny + 1), device=device) * params.depth.to(device)
# H[0] = 1 * torch.exp(-((X-X.max()//2)**2 + (Y-Y.max()//2)**2) / (2 * 50000**2))
#
Wx,Wy,Pa = generate_wind(X,Y,params)
Ws = torch.sqrt(Wx**2 + Wy**2)
#
# Wx = np.ones((NT,X.shape[0],X.shape[1]))*0 + 0
# Wy = np.ones((NT,X.shape[0],X.shape[1]))*0 
# Wx = torch.from_numpy(Wx)
# Wy = torch.from_numpy(Wy)
# Pa = Wx * 0 + 1000

eta_tensor = torch.tensor(eta_array, dtype=torch.float32).to(device)
u_tensor = torch.tensor(u_array, dtype=torch.float32).to(device)
v_tensor = torch.tensor(v_array, dtype=torch.float32).to(device)

target_size = eta_tensor.shape[1:]  

u_resized = F.interpolate(u_tensor.unsqueeze(1), 
                          size=target_size, mode='bilinear', align_corners=False).squeeze(1)
v_resized = F.interpolate(v_tensor.unsqueeze(1), 
                          size=target_size, mode='bilinear', align_corners=False).squeeze(1)


X_tensor = torch.stack([eta_tensor, u_resized, v_resized], dim=1)  
Y_tensor = torch.tensor(manning_array, dtype=torch.float32)        

eta_array = torch.tensor(eta_array, dtype=torch.float32)
u_array = torch.tensor(u_array, dtype=torch.float32)
v_array = torch.tensor(v_array, dtype=torch.float32)

input_channel = 3
model = CNN1(shapeX=input_channel).to(device)
model.float()  
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

torch.autograd.set_detect_anomaly(True)

# 优化后的训练循环：直接利用整个训练序列（X_tensor）进行多步模拟
num_epochs = 200
eta_list = []
u_list = []
v_list = []
train_loss_history = []
manning_list = []
Fx_list = []
Fy_list = []
loss_list = []

#manning = torch.rand((params.Nx+1, params.Ny+1)).requires_grad_(True).to(device)
for epoch in range(num_epochs):
    model.train()
    
    epoch_loss_sum = 0.0  # 用于累加当前 epoch 内所有时间步的 loss
    H_train = list()
    M_train = list()
    N_train = list()
    current_input = X_tensor[0].unsqueeze(0).to(device).float()
    M = torch.zeros((2, params.Nx, params.Ny+1)).to(device)
    N = torch.zeros((2, params.Nx+1, params.Ny)).to(device)
    H = torch.zeros((2, params.Nx+1, params.Ny+1)).to(device)
    for t in range(params.NT):
        # 保存当前时间步的初始状态
        H_init = H.clone()
        M_init = M.clone()
        N_init = N.clone()
        inner_steps = 20
        for i in range(inner_steps):
            # 使用固定的初始状态 H_init, M_init, N_init 进行前向传播
            manning = model(current_input).squeeze(0).squeeze(0)
            H_update = mass_cartesian_torch(H_init, Z, M_init, N_init, params)
            M_update, N_update, Fx, Fy = momentum_nonlinear_cartesian_torch(
                H_update, Z, M_init, N_init, Wx[t], Wy[t], Pa[t], params, manning)
            
            # 计算 loss，这里确保输入和目标尺寸匹配
            loss_eta = criterion(H_update[1], eta_array[t].to(device))
            loss_u = criterion(M_update[1], u_array[t].to(device))
            loss_v = criterion(N_update[1], v_array[t].to(device))
            loss = loss_eta + loss_u + loss_v
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 更新状态，构造新的 current_input
        H_update = mass_cartesian_torch(H_init, Z, M_init, N_init, params)
        M_update, N_update, Fx, Fy = momentum_nonlinear_cartesian_torch(
            H_update, Z, M_init, N_init, Wx[t], Wy[t], Pa[t], params, manning)
        
        H[0] = H_update[1].detach()
        M[0] = M_update[1].detach()
        N[0] = N_update[1].detach()
        
        target_size = H_update[1].shape
        M_resized = F.interpolate(M_update[1].unsqueeze(0).unsqueeze(0),
                                  size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        N_resized = F.interpolate(N_update[1].unsqueeze(0).unsqueeze(0),
                                  size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        current_state = torch.stack([H[0], M_resized, N_resized], dim=0)
        current_input = current_state.unsqueeze(0).detach()
        
        # 累加当前时间步的 loss
        epoch_loss_sum += loss.item()
        
        if epoch == num_epochs - 1:
            # 记录变量（只在最后一个 epoch 记录）
            eta_list.append(dd(H_update[1]))
            u_list.append(dd(M_update[1]))
            v_list.append(dd(N_update[1]))
            manning_list.append(manning.detach().cpu().numpy())
            Fx_list.append(Fx.detach().cpu().numpy())
            Fy_list.append(Fy.detach().cpu().numpy())
            
            np.save('eta_list_train_manning.npy', np.array(eta_list))
            np.save('u_list_train_manning.npy', np.array(u_list))
            np.save('v_list_train_manning.npy', np.array(v_list))
            np.save('manning_list_train.npy', np.array(manning_list))
            np.save('Fx_list_train_manning.npy', np.array(Fx_list))
            np.save('Fy_list_train_manning.npy', np.array(Fy_list))
    
    # 计算该 epoch 内所有时间步的平均 loss，并保存
    epoch_avg_loss = epoch_loss_sum / params.NT
    train_loss_history.append(epoch_avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_avg_loss:.15f}")

# 保存整个训练的平均 loss 历史
np.save('train_loss_history_manning.npy', np.array(train_loss_history))
torch.save(model.state_dict(), 'model_checkpoint_manning.pth')
print("Training complete and data saved.")
    

