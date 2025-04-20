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
from dynamics_train import mass_cartesian_torch,momentum_nonlinear_cartesian_torch
import torch.nn as nn
import torch.optim as optim
from CNN_manning_diff import CNN1
import torch
import time  # 在文件头部导入
#from radam import RAdam  # 或 pip install radam，具体看第三方库
# from torchviz import make_dot      
def tic():
    global _start_time
    _start_time = time.time()

def toc():
    elapsed_time = time.time() - _start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")


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
device = 'cuda'
print(device)

case = 'large'
exec(f'params = Params_{case}(device)')


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
manning_distribution = np.load(f'manning_{case}_distribution.npy')
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

#%%
input_channel = 3
model = CNN1(shapeX=input_channel).to(device)
model.float()  
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

torch.autograd.set_detect_anomaly(True)

# 优化后的训练循环：直接利用整个训练序列（X_tensor）进行多步模拟
num_epochs = 10
inner_steps = 10
params.NT = 50 ##### overall time = num_epochs * inner_steps * NT = 5000
eta_list = []
u_list = []
v_list = []
train_loss_history = []
manning_list = []
Fx_list = []
Fy_list = []
loss_list = []

#
#mask = torch.zeros_like(eta_array[0], dtype=bool)
mask = torch.zeros(eta_array[0].shape, dtype=bool).to(device)
mask.view(-1)[torch.randperm(eta_array[0].numel())[:10]] = True


#manning = torch.rand((params.Nx+1, params.Ny+1)).requires_grad_(True).to(device)
last_epoch_H_list = []
last_epoch_M_list = []
last_epoch_N_list = []
last_epoch_manning_list = []
#%%
for epoch in range(num_epochs):
    model.train()
    #print(f'Start Training {epoch}/{num_epochs}')
    epoch_loss_sum = 0.0  # 累加当前 epoch 内所有时间步的 loss
    current_input = X_tensor[0].unsqueeze(0).to(device).float()
    
    # 重新初始化状态，每个 epoch 内从零开始
    M = torch.zeros((2, params.Nx, params.Ny+1)).to(device)
    N = torch.zeros((2, params.Nx+1, params.Ny)).to(device)
    H = torch.zeros((2, params.Nx+1, params.Ny+1)).to(device)
    
    H0 = H.clone()
    M0 = M.clone()
    N0 = N.clone()
    
    H1 = mass_cartesian_torch(H0, Z, M0, N0, params)
    
    for t in range(params.NT):
        # 保存当前时间步的初始状态
        for i in range(inner_steps):
            current_input = torch.stack([H1[1], u2rho(M0[0],'expand'), v2rho(N0[0],'expand')], dim=0).unsqueeze(0) 
            manning = model(current_input).squeeze(0).squeeze(0)
            
            M1, N1, Fx, Fy = momentum_nonlinear_cartesian_torch(
                H1, Z, M0, N0, Wx[t], Wy[t], Pa[t], params, manning)
            
            H2 = mass_cartesian_torch(H1, Z, M1, N1, params)
            
            scale = 1
            tmp1 = torch.masked_select(H2[1]* scale, mask)
            tmp2 = torch.masked_select(eta_array[t].to(device)* scale, mask)
            loss_eta = criterion(tmp1, tmp2)
            
            #loss_eta = criterion(H2[1]* scale, eta_array[t].to(device))
            #loss_u = criterion(M_update[1]* scale, u_array[t].to(device)* scale)
            #loss_v = criterion(N_update[1]* scale, v_array[t].to(device)* scale)
    
            loss = loss_eta
            print(f"Before Loss: {loss.item():.8e}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        
        # 更新状态，用于下一个时间步
        H1[0] = H2[1].detach()
        M0[0] = M1[1].detach()
        N0[0] = N1[1].detach()
        
        # for i in range(inner_steps):
        #     #print(i)
        #     #tic()
        #     manning = model(current_input).squeeze(0).squeeze(0)
        #     H_update = mass_cartesian_torch(H_init, Z, M_init, N_init, params)
        #     M_update, N_update, Fx, Fy = momentum_nonlinear_cartesian_torch(
        #         H_update, Z, M_init, N_init, Wx[t], Wy[t], Pa[t], params, manning)
            
        #     scale = 1
        #     tmp1 = torch.masked_select(H_update[1]* scale, mask)
        #     tmp2 = torch.masked_select(eta_array[t].to(device)* scale, mask)
        #     loss_eta = criterion(tmp1, tmp2)
        #     #loss_u = criterion(M_update[1]* scale, u_array[t].to(device)* scale)
        #     #loss_v = criterion(N_update[1]* scale, v_array[t].to(device)* scale)

        #     loss = loss_eta
            
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
            
        #     #toc()
        #     #print(f"Inner loop")
        
        # # 用最新的模型参数计算一次更新（不进行梯度更新）
        # H_update = mass_cartesian_torch(H_init, Z, M_init, N_init, params)
        # M_update, N_update, Fx, Fy = momentum_nonlinear_cartesian_torch(
        #     H_update, Z, M_init, N_init, Wx[t], Wy[t], Pa[t], params, manning)
        
        # 更新状态，用于下一个时间步
#         H[0] = H_update[1].detach()
#         M[0] = M_update[1].detach()
#         N[0] = N_update[1].detach()
        
#         # 对 M_update、N_update 做插值（保证与 H_update 一致）
#         target_size = H_update[1].shape
#         M_resized = F.interpolate(M_update[1].unsqueeze(0).unsqueeze(0),
#                                   size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
#         N_resized = F.interpolate(N_update[1].unsqueeze(0).unsqueeze(0),
#                                   size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
#         current_state = torch.stack([H[0], M_resized, N_resized], dim=0)
#         current_input = current_state.unsqueeze(0).detach()
        
#         # 计算当前时间步的最终 loss（用最新的状态 H_update、M_update、N_update 计算）
        
#         final_loss_eta = criterion(H_update[1]* scale, eta_array[t].to(device)* scale)
#         final_loss_u = criterion(M_update[1]* scale, u_array[t].to(device)* scale)
#         final_loss_v = criterion(N_update[1]* scale, v_array[t].to(device)* scale)
#         final_loss = final_loss_eta + final_loss_u + final_loss_v
#         epoch_loss_sum += final_loss.item()
        
#         print(f"Epoch { (epoch+1+ t/params.NT) :.2f}/{num_epochs}, Train Loss: {epoch_loss_sum:.5e}")
        
#         # 如果是最后一个 epoch，则保存每个时间步的最终状态和 loss
#         if epoch == num_epochs - 1:
#             last_epoch_H_list.append(dd(H_update[1]))
#             last_epoch_M_list.append(dd(M_update[1]))
#             last_epoch_N_list.append(dd(N_update[1]))
#             last_epoch_manning_list.append(dd(manning))
    
#     # 计算当前 epoch 内所有时间步的平均 loss，并保存到总的 loss 历史中
#     epoch_avg_loss = epoch_loss_sum / params.NT
#     train_loss_history.append(epoch_avg_loss)
#     plot(train_loss_history);show()
#     print(f"Epoch {epoch+1} Finish, Train Loss: {epoch_avg_loss:.5e}")

# # 保存最后一个 epoch 的每个时间步状态和 loss
# np.save('eta_list_train_manning.npy', np.array(last_epoch_H_list))
# np.save('u_list_train_manning.npy', np.array(last_epoch_M_list))
# np.save('v_list_train_manning.npy', np.array(last_epoch_N_list))
# last_epoch_manning_list
# np.save('manning_list_train_manning.npy', np.array(last_epoch_manning_list))
# # 保存整个 epoch 平均 loss 历史
# np.save('train_loss_history_manning.npy', np.array(train_loss_history))
# torch.save(model.state_dict(), 'model_checkpoint_manning.pth')
# print("Training complete and data saved.")
    

