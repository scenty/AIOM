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
from All_params import Params,Params_tsunami
from dynamics import mass_cartesian_torch, momentum_nonlinear_cartesian_torch
import torch.nn as nn
import torch.optim as optim
from CNN_manning import CNN1
import torch
#time module
import time
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

def Z_slope(X,Y):
    hmax = 100.0  # 离岸水深（米）
    hmin = 1.0   # 近岸水深（米）
    x_start = 200000.0  # 斜坡过渡带长度（公里）
       
    # 线性过渡公式（自动截断在[1, 100]米范围内）
    Z = hmax - (X - x_start ) / (x.max() - x_start) * ( hmax - hmin )
    Z[Z>100] = 100
    return Z

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)
    params = Params(device)

    # 读取 eta 数据
    ds_eta = xr.open_dataset('out_eta.nc')
    eta_array = ds_eta['eta'].values  # 或者直接使用 ds_eta.eta
    
    # 读取 u 数据
    ds_u = xr.open_dataset('out_u.nc')
    u_array = ds_u['u'].values  # 或者直接使用 ds_u.u
    
    # 读取 v 数据
    ds_v = xr.open_dataset('out_v.nc')
    v_array = ds_v['v'].values  # 或者直接使用 ds_v.v
 
    dt = params.dt
    NT = params.NT
    
    x = torch.linspace(0, params.Lx, params.Nx + 1)
    y = torch.linspace(0, params.Ly, params.Ny + 1)
    X, Y = torch.meshgrid(x, y)
    X = X.to(device)
    Y = Y.to(device)
    manning_array = np.zeros((v_array.shape[0], params.Nx+1, params.Ny+1))

    
    M = torch.zeros((2, params.Nx, params.Ny+1))
    N = torch.zeros((2, params.Nx+1, params.Ny))
    H = torch.zeros((2, params.Nx+1, params.Ny+1))
    # 确保将 params.depth 转移到正确的设备上
    Z = Z_slope(X,Y).to(device)
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
    
    num_time_steps_train = 200        
    X_tensor = torch.tensor(np.stack([eta_array, u_array, v_array], axis=1), dtype=torch.float32)
    Y_tensor = torch.tensor(manning_array, dtype=torch.float32)
    eta_array = torch.tensor(eta_array, dtype=torch.float32)
    
    input_channel = 3
    model = CNN1(shapeX=input_channel).to(device)
    model.float()  
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    torch.autograd.set_detect_anomaly(True)
    
    # 优化后的训练循环：直接利用整个训练序列（X_tensor）进行多步模拟
    num_epochs = 100
    chunk_size = 10  # 每隔多少步反传一次
    eta_list = []
    u_list = []
    v_list = []
    train_loss_history = []
    manning_list = []

    loss_list = []

    # 在训练循环中确保数据也转移到 GPU
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        acc_loss = 0.0      # 累计 chunk 内 loss
        # 训练数据也需要转移到 GPU
        H = torch.zeros((2, params.Nx+1, params.Ny+1)).clone().detach().requires_grad_(True).to(device)
        M = torch.zeros((2, params.Nx, params.Ny+1)).clone().detach().requires_grad_(True).to(device)
        N = torch.zeros((2, params.Nx+1, params.Ny)).clone().detach().requires_grad_(True).to(device)

        for t in range(params.NT):
            current_input = X_tensor[t].unsqueeze(0).to(device).float()
            manning = model(current_input).squeeze(0)
            
            
            H_update = mass_cartesian_torch(H, Z, M, N, params)
            M_update, N_update = momentum_nonlinear_cartesian_torch(H_update, Z, M, N, Wx[t], Wy[t], Pa[t], params, manning)

            scale = 1
            # loss_t = criterion(H_update[1] * scale, eta_array[t].to(device) * scale)
            loss_t = criterion(H_update[1] * scale, eta_array[t].to(device))
            acc_loss += loss_t

    
            if (t + 1) % chunk_size == 0 or (t == num_time_steps_train - 1):
                print('Backward and update the net')
                optimizer.zero_grad()
                acc_loss.backward()
                optimizer.step()
    
                running_loss += acc_loss.item()
                # 更新 H, M, N 的状态
                H = torch.stack((H_update[1].clone(), H_update[1].clone()), dim=0).detach().requires_grad_(True).to(device)
                M = torch.stack((M_update[1].clone(), M_update[1].clone()), dim=0).detach().requires_grad_(True).to(device)
                N = torch.stack((N_update[1].clone(), N_update[1].clone()), dim=0).detach().requires_grad_(True).to(device)
                acc_loss = 0.0
            else:
                print('Stack simulation')
                H = torch.stack((H_update[1].clone(), H_update[1].clone()), dim=0)
                M = torch.stack((M_update[1].clone(), M_update[1].clone()), dim=0)
                N = torch.stack((N_update[1].clone(), N_update[1].clone()), dim=0)
    
            if epoch == num_epochs - 1:
                # 记录变量
                eta_list.append(dd(H_update[1]))
                u_list.append(dd(M_update[1]))
                v_list.append(dd(N_update[1]))
                manning_list.append(manning.detach().cpu().numpy())
                
                # 保存数据到文件
                np.save('eta_list_train_manning.npy', np.array(eta_list))  # 转换为 numpy 数组并保存
                np.save('u_list_train_manning.npy', np.array(u_list))      # 转换为 numpy 数组并保存
                np.save('v_list_train_manning.npy', np.array(v_list))      # 转换为 numpy 数组并保存
                np.save('manning_list_train.npy', np.array(manning_list))    # 转换为 numpy 数组并保存

                
                # 保存损失历史
                np.save('train_loss_history_manning.npy', np.array(train_loss_history))  # 转换为 numpy 数组并保存
                
                # 保存模型权重（state_dict）
                torch.save(model.state_dict(), 'model_checkpoint_manning.pth')
                
                print("Training complete and data saved.")
    
        avg_train_loss = running_loss / num_time_steps_train
        train_loss_history.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.15f}")
