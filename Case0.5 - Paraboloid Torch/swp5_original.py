# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 19:48:17 2025

@author: ASUS
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from CNN_noip import CNN1  
#from torch.utils.data import TensorDataset, DataLoader   # 不再使用 DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##############################################################################
# 0. 全局设置：双精度
##############################################################################
torch.set_default_dtype(torch.float64)

##############################################################################
# 1. 网格、时间、物理参数
##############################################################################
Lx, Ly = 8000.0, 8000.0
Nx, Ny = 75, 75
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

x_t = torch.linspace(-Lx/2, Lx/2, Nx, dtype=torch.float64)
y_t = torch.linspace(-Ly/2, Ly/2, Ny, dtype=torch.float64)
X, Y = torch.meshgrid(x_t, y_t)  # shape (Nx, Ny)

g = 9.81 #TODO(Lu) 9.8

dry_limit = 1e5
MinWaterDepth = 1e-4
FrictionDepthLimit = 5e-3
cfl_factor = 0.9  # 安全系数

dt = 5

NT = 200  # 最大步数
num_time_steps = int(NT * 0.8)

##############################################################################
# 2. 床面与初始水面
##############################################################################
def bed_elevation_torch(X, Y, D0, L):
    r = torch.sqrt(X**2 + Y**2)
    z = -D0*(1.0 - r**2/(L**2))
    return z

def stage_init_torch(X, Y, D0, L, A, omega, t):
    """
    对应您 NumPy 版本的 stage_init(x, y):
      w = D0*((sqrt(1 - A^2))/(1 - A*cos(omega*t)) - 1.0
              - r^2/(L^2)*((1 - A^2)/((1 - A*cos(omega*t))^2) - 1.0))
      if w < z: w = z
    其中 z = bed_elevation_torch_npStyle(X, Y, D0, L)
    """
    z = bed_elevation_torch(X, Y, D0, L)
    r = torch.sqrt(X**2 + Y**2)
    # 逐点计算 w
    w = D0*(
        (torch.sqrt(1. - A*A))/(1. - A*torch.cos(omega*t)) - 1.
        - r**2/(L**2)*(
            (1. - A*A)/((1. - A*torch.cos(omega*t))**2) - 1.
        )
    )
    # 若 w < z，则 w = z
    w = torch.max(w, z)
    return w


D0 = torch.tensor(1.0, dtype=torch.float64)
L = torch.tensor(2500.0, dtype=torch.float64)
R0 = torch.tensor(2000.0, dtype=torch.float64)
t = 0.0
A = (L**4 - R0**4) / (L**4 + R0**4)
omega = 2. / L * torch.sqrt(2. * g * D0)

Z = -bed_elevation_torch(X, Y, D0, L)
# 保持 double 并放到 device 上
Z = torch.tensor(Z, device=device, dtype=torch.float64)
H_init_0 = stage_init_torch(X, Y, D0, L, A, omega, t)

##############################################################################
# 3. 分配张量：H, M, N
##############################################################################
H = torch.zeros((2, Nx, Ny), dtype=torch.float64)
H[0] = H_init_0.clone()

M = torch.zeros((2, Nx-1, Ny), dtype=torch.float64)
N = torch.zeros((2, Nx, Ny-1), dtype=torch.float64)

H_init = H.clone()
M_init = M.clone()
N_init = N.clone()

##############################################################################
# 4. 辅助函数：check_flux_direction
##############################################################################
def check_flux_direction_torch(z1, z2, h1, h2, dry_limit):
    device = z1.device
    h1_out = h1.clone()
    h2_out = h2.clone()
    flux_sign = torch.zeros_like(z1, dtype=torch.int64, device=device)

    cond1 = (z1 <= -dry_limit) | (z2 <= -dry_limit)
    flux_sign = torch.where(cond1, torch.tensor(999, dtype=torch.int64, device=device), flux_sign)

    cond2 = ((z1 + h1_out) <= 0) | ((z2 + h2_out) <= 0)
    cond2a = ((z1 + h1_out) <= 0) & ((z2 + h2_out) <= 0)
    cond2b = (z1 > z2) & (z1 + h1_out > 0) & ((z2 + h1_out) <= 0)
    cond2c = (z2 > z1) & (z2 + h2_out > 0) & ((z1 + h2_out) <= 0)
    cond2_any = cond2 & (cond2a | cond2b | cond2c)
    flux_sign = torch.where(cond2_any, torch.tensor(999, dtype=torch.int64, device=device), flux_sign)

    cond3 = (z1 <= z2) & (z1 + h1_out > 0) & ((z1 + h2_out) <= 0)
    flux_sign = torch.where(cond3, torch.tensor(1, dtype=torch.int64, device=device), flux_sign)
    h2_out = torch.where(cond3, -z1, h2_out)

    cond4 = (z1 >= z2) & ((z2 + h2_out) <= 0) & (z2 + h1_out > 0)
    flux_sign = torch.where(cond4, torch.tensor(2, dtype=torch.int64, device=device), flux_sign)
    h2_out = torch.where(cond4, -z2, h2_out)

    cond5 = (z2 <= z1) & (z2 + h2_out > 0) & ((z2 + h1_out) <= 0)
    flux_sign = torch.where(cond5, torch.tensor(-1, dtype=torch.int64, device=device), flux_sign)
    h1_out = torch.where(cond5, -z2, h1_out)

    cond6 = (z2 >= z1) & ((z1 + h1_out) <= 0) & (z1 + h2_out > 0)
    flux_sign = torch.where(cond6, torch.tensor(-2, dtype=torch.int64, device=device), flux_sign)
    h1_out = torch.where(cond6, -z1, h1_out)

    return flux_sign, h1_out, h2_out

##############################################################################
# 5. 重构流深 (D_M, D_N)
##############################################################################
def reconstruct_flow_depth_torch(H, Z, M, N, dry_limit):
    # x方向通量
    z1_M = Z[:-1, :]
    z2_M = Z[1:, :]
    h1_M = H[1, :-1, :]
    h2_M = H[1, 1:, :]
    flux_sign_M, h1u_M, h2u_M = check_flux_direction_torch(z1_M, z2_M, h1_M, h2_M, dry_limit)

    m0 = M[0]
    D_M = torch.zeros_like(m0)
    mask_999_M = (flux_sign_M == 999)
    mask_0_M = (flux_sign_M == 0)
    mask_non0_M = (~mask_0_M) & (~mask_999_M)
    cond_m0pos = (m0 >= 0)

    D_M = torch.where(mask_0_M & cond_m0pos, 0.5*(z1_M + z2_M) + h1u_M, D_M)
    D_M = torch.where(mask_0_M & (~cond_m0pos), 0.5*(z1_M + z2_M) + h2u_M, D_M)
    D_M = torch.where(mask_non0_M, 0.5*(z1_M + z2_M) + torch.maximum(h1u_M, h2u_M), D_M)

    # y方向通量
    z1_N = Z[:, :-1]
    z2_N = Z[:, 1:]
    h1_N = H[1, :, :-1]
    h2_N = H[1, :, 1:]
    flux_sign_N, h1u_N, h2u_N = check_flux_direction_torch(z1_N, z2_N, h1_N, h2_N, dry_limit)

    n0 = N[0]
    D_N = torch.zeros_like(n0)
    mask_999_N = (flux_sign_N == 999)
    mask_0_N = (flux_sign_N == 0)
    mask_non0_N = (~mask_0_N) & (~mask_999_N)
    cond_n0pos = (n0 >= 0)

    D_N = torch.where(mask_0_N & cond_n0pos, 0.5*(z1_N + z2_N) + h1u_N, D_N)
    D_N = torch.where(mask_0_N & (~cond_n0pos), 0.5*(z1_N + z2_N) + h2u_N, D_N)
    D_N = torch.where(mask_non0_N, 0.5*(z1_N + z2_N) + torch.maximum(h1u_N, h2u_N), D_N)

    return D_M, D_N

##############################################################################
# 6. 质量方程 (mass_cartesian_torch)
##############################################################################
def mass_cartesian_torch(H, Z, M, N, dt):
    H0 = H[0]
    Nx_, Ny_ = H0.shape

    # 构造 m_right, m_left
    M0 = M[0]
    m_right = torch.zeros((Nx_, Ny_), dtype=M0.dtype, device=M0.device)
    m_right[:Nx_-1, :] = M0
    m_left = torch.zeros_like(m_right)
    m_left[1:Nx_, :] = M0

    # 构造 n_up, n_down
    N0 = N[0]
    n_up = torch.zeros((Nx_, Ny_), dtype=N0.dtype, device=N0.device)
    n_up[:, :Ny_-1] = N0
    n_down = torch.zeros_like(n_up)
    n_down[:, 1:Ny_] = N0

    CC1_ = dt/dx
    CC2_ = dt/dy

    H1 = H0 - CC1_*(m_right - m_left) - CC2_*(n_up - n_down)

    # 干湿修正
    mask_deep = (Z <= -dry_limit)
    H1 = torch.where(mask_deep, torch.zeros_like(H1), H1)

    ZH0 = Z + H0
    ZH1 = Z + H1
    cond = (Z < dry_limit) & (Z > -dry_limit)

    wet_to_dry = cond & (ZH0>0) & ((H1-H0)<0) & (ZH1<=MinWaterDepth)
    c1 = wet_to_dry & (Z>0)
    c2 = wet_to_dry & (Z<=0)
    H1 = torch.where(c1, -Z, H1)
    H1 = torch.where(c2, torch.zeros_like(H1), H1)

    cond_dry = cond & (ZH0<=0)
    c3 = cond_dry & ((H1-H0)>0)
    H1 = torch.where(c3, H1 - H0 - Z, H1)
    c4 = cond_dry & ((H1-H0)<=0) & (Z>0)
    c5 = cond_dry & ((H1-H0)<=0) & (Z<=0)
    H1 = torch.where(c4, -Z, H1)
    H1 = torch.where(c5, torch.zeros_like(H1), H1)

    new_H = torch.stack((H0, H1), dim=0)
    return new_H

##############################################################################
# 7. 动量方程 (momentum_nonlinear_cartesian)
##############################################################################
def momentum_nonlinear_cartesian(H, Z, M, N, D_M, D_N, dt, Fx, Fy):
    Dlimit = 1e-3

    # M 分量
    m0 = M[0]
    Nx_m, Ny_m = m0.shape
    D0 = D_M
    if Nx_m > 1:
        m1 = torch.cat((torch.zeros((1, Ny_m), dtype=m0.dtype, device=m0.device), m0[:-1, :]), dim=0)
        m2 = torch.cat((m0[1:, :], torch.zeros((1, Ny_m), dtype=m0.dtype, device=m0.device)), dim=0)
        D1 = torch.cat((torch.zeros((1, Ny_m), dtype=D0.dtype, device=D0.device), D0[:-1, :]), dim=0)
        D2 = torch.cat((D0[1:, :], torch.zeros((1, Ny_m), dtype=D0.dtype, device=D0.device)), dim=0)
    else:
        m1 = m0.clone()
        m2 = m0.clone()
        D1 = D0.clone()
        D2 = D0.clone()

    z1_M = Z[:-1, :]
    z2_M = Z[1:, :]
    h1_M = H[1, :-1, :]
    h2_M = H[1, 1:, :]
    flux_sign_M, h1u_M, h2u_M = check_flux_direction_torch(z1_M, z2_M, h1_M, h2_M, dry_limit)

    cond_m0pos = (m0 >= 0)
    maskD0 = (D0 > Dlimit)
    maskD1 = (D1 > Dlimit)
    maskD2 = (D2 > Dlimit)

    MU1 = torch.where(cond_m0pos & maskD1, m1**2 / torch.clamp(D1, min=MinWaterDepth), torch.zeros_like(m0))
    MU2 = torch.where(cond_m0pos & maskD0, m0**2 / torch.clamp(D0, min=MinWaterDepth), torch.zeros_like(m0))
    MU1 = torch.where(~cond_m0pos & maskD0, m0**2 / torch.clamp(D0, min=MinWaterDepth), MU1)
    MU2 = torch.where(~cond_m0pos & maskD2, m2**2 / torch.clamp(D2, min=MinWaterDepth), MU2)

    phi_M = 1.0 - (dt/dx)*torch.min(
        torch.abs(m0 / torch.clamp(D0, min=MinWaterDepth)),
        torch.sqrt(g * torch.clamp(D0, min=MinWaterDepth))
    )

    M_val = phi_M*m0 + 0.5*(1.0 - phi_M)*(m1 + m2) - (dt/dx)*(MU2 - MU1) - (dt*g/dx)*D0*(h2u_M - h1u_M)
    mask_fs999_M = (flux_sign_M == 999)
    M_val = torch.where(mask_fs999_M, torch.zeros_like(M_val), M_val)

    friction_mask = D0 > FrictionDepthLimit
    friction_term_M = torch.where(friction_mask, dt * Fx[:Nx_m, :], torch.zeros_like(Fx[:Nx_m, :]))
    M_val = M_val - friction_term_M

    M_new = torch.stack((M[0], M_val), dim=0)

    # N 分量
    n0 = N[0]
    Nx_n, Ny_n = n0.shape
    D0N = D_N
    if Ny_n > 1:
        n1 = torch.cat((torch.zeros((Nx_n, 1), dtype=n0.dtype, device=n0.device), n0[:, :-1]), dim=1)
        n2 = torch.cat((n0[:, 1:], torch.zeros((Nx_n, 1), dtype=n0.dtype, device=n0.device)), dim=1)
        D1N = torch.cat((torch.zeros((Nx_n, 1), dtype=D0N.dtype, device=D0N.device), D0N[:, :-1]), dim=1)
        D2N = torch.cat((D0N[:, 1:], torch.zeros((Nx_n, 1), dtype=D0N.dtype, device=D0N.device)), dim=1)
    else:
        n1 = n0.clone()
        n2 = n0.clone()
        D1N = D0N.clone()
        D2N = D0N.clone()

    z1_N = Z[:, :-1]
    z2_N = Z[:, 1:]
    h1_N = H[1, :, :-1]
    h2_N = H[1, :, 1:]
    flux_sign_N, h1u_N, h2u_N = check_flux_direction_torch(z1_N, z2_N, h1_N, h2_N, dry_limit)

    cond_n0pos = (n0 >= 0)
    maskD0N = (D0N > Dlimit)
    maskD1N = (D1N > Dlimit)
    maskD2N = (D2N > Dlimit)

    NV1 = torch.where(cond_n0pos & maskD1N, n1**2 / torch.clamp(D1N, min=MinWaterDepth), torch.zeros_like(n0))
    NV2 = torch.where(cond_n0pos & maskD0N, n0**2 / torch.clamp(D0N, min=MinWaterDepth), torch.zeros_like(n0))
    NV1 = torch.where(~cond_n0pos & maskD0N, n0**2 / torch.clamp(D0N, min=MinWaterDepth), NV1)
    NV2 = torch.where(~cond_n0pos & maskD2N, n2**2 / torch.clamp(D2N, min=MinWaterDepth), NV2)

    phi_N = 1.0 - (dt/dy)*torch.min(
        torch.abs(n0 / torch.clamp(D0N, min=MinWaterDepth)),
        torch.sqrt(g * torch.clamp(D0N, min=MinWaterDepth))
    )

    N_val = phi_N*n0 + 0.5*(1.0 - phi_N)*(n1 + n2) - (dt/dy)*(NV2 - NV1) - (dt*g/dy)*D0N*(h2u_N - h1u_N)
    mask_fs999_N = (flux_sign_N == 999)
    N_val = torch.where(mask_fs999_N, torch.zeros_like(N_val), N_val)

    if Ny_n > 1:
        friction_mask_N = D0N > FrictionDepthLimit
        friction_term_N = torch.where(friction_mask_N[:, :-1],
                                      dt * Fy[:, :Ny_n-1],
                                      torch.zeros_like(Fy[:, :Ny_n-1]))
        N_val_updated = torch.cat((N_val[:, :-1] - friction_term_N, N_val[:, -1:]), dim=1)
    else:
        N_val_updated = N_val

    N_new = torch.stack((N[0], N_val_updated), dim=0)
    return M_new, N_new

##############################################################################
# 8. 自适应 CFL：计算最大波速 -> 调整 dt
##############################################################################
def compute_dt_cfl(H, M, N, Z, dx, dy, g, cfl_factor=0.9):
    Z = Z.to(device=device, dtype=torch.float64)
    H_cur = H[1].to(Z.device)
    water_depth = torch.clamp(H_cur - Z, min=0.0)
    eps = 1e-14

    max_depth = torch.max(water_depth)
    wave_speed = torch.sqrt(g*max_depth).item()

    M0_absmax = torch.max(torch.abs(M[0]))
    N0_absmax = torch.max(torch.abs(N[0]))
    if max_depth > 0:
        u_max = (M0_absmax / max_depth).item()
        v_max = (N0_absmax / max_depth).item()
    else:
        u_max, v_max = 0.0, 0.0

    c_x = (u_max + wave_speed)
    c_y = (v_max + wave_speed)
    if c_x < 1e-14 and c_y < 1e-14:
        return 1e6

    dt_cfl_x = dx/(c_x + eps)
    dt_cfl_y = dy/(c_y + eps)
    dt_cfl = cfl_factor*min(dt_cfl_x, dt_cfl_y)
    return dt_cfl

##############################################################################
# 9. 边界处理 (apply_closed_boundary)
##############################################################################
def apply_closed_boundary(H, M, N, Z):
    Nx_, Ny_ = Z.shape
    layer = 1

    H_layer = H[layer].clone()
    threshold = 0.005
    stage = Z - H_layer
    mask_bedpos = (torch.abs(stage) < threshold)

    if (H_layer[mask_bedpos] < 0.05).any():
        H_layer = torch.where(stage < -0.2, torch.zeros_like(H_layer), H_layer)

        left_boundary = torch.clamp(Z[0, :] + H_layer[0, :],
                                    min=Z[0, :],
                                    max=torch.zeros_like(Z[0, :])) - Z[0, :]
        right_boundary = torch.clamp(Z[-1, :] + H_layer[-1, :],
                                     min=Z[-1, :],
                                     max=torch.zeros_like(Z[-1, :])) - Z[-1, :]
        bottom_boundary = torch.clamp(Z[:, 0] + H_layer[:, 0],
                                      min=Z[:, 0],
                                      max=torch.zeros_like(Z[:, 0])) - Z[:, 0]
        top_boundary = torch.clamp(Z[:, -1] + H_layer[:, -1],
                                   min=Z[:, -1],
                                   max=torch.zeros_like(Z[:, -1])) - Z[:, -1]

        Nx = Nx_
        Ny = Ny_
        rows = torch.arange(Nx, device=H_layer.device).view(Nx, 1).expand(Nx, Ny)
        cols = torch.arange(Ny, device=H_layer.device).view(1, Ny).expand(Nx, Ny)

        H_layer_updated = H_layer
        H_layer_updated = torch.where(rows == 0, left_boundary.unsqueeze(0).expand(Nx, Ny), H_layer_updated)
        H_layer_updated = torch.where(rows == Nx - 1, right_boundary.unsqueeze(0).expand(Nx, Ny), H_layer_updated)
        H_layer_updated = torch.where(cols == 0, bottom_boundary.unsqueeze(1).expand(Nx, Ny), H_layer_updated)
        H_layer_updated = torch.where(cols == Ny - 1, top_boundary.unsqueeze(1).expand(Nx, Ny), H_layer_updated)
    else:
        H_layer_updated = H_layer

    M_layer = M[layer].clone()
    M_Nx, M_Ny = M_layer.shape
    M_rows = torch.arange(M_Nx, device=M_layer.device).view(M_Nx, 1).expand(M_Nx, M_Ny)
    M_layer_updated = M_layer
    if M_Nx > 1:
        M_layer_updated = torch.where(M_rows == 0, torch.zeros_like(M_layer_updated), M_layer_updated)
        M_layer_updated = torch.where(M_rows == M_Nx - 1, torch.zeros_like(M_layer_updated), M_layer_updated)

    N_layer = N[layer].clone()
    N_Nx, N_Ny = N_layer.shape
    N_cols = torch.arange(N_Ny, device=N_layer.device).view(1, N_Ny).expand(N_Nx, N_Ny)
    N_layer_updated = N_layer
    if N_Ny > 1:
        N_layer_updated = torch.where(N_cols == 0, torch.zeros_like(N_layer_updated), N_layer_updated)
        N_layer_updated = torch.where(N_cols == N_Ny - 1, torch.zeros_like(N_layer_updated), N_layer_updated)

    H_new = torch.stack((H[0], H_layer_updated), dim=0)
    M_new = torch.stack((M[0], M_layer_updated), dim=0)
    N_new = torch.stack((N[0], N_layer_updated), dim=0)
    return H_new, M_new, N_new

##############################################################################
# 10. 主时间步进循环 & 训练部分
##############################################################################
eta_array = np.load("eta_pt_array_1.npy", allow_pickle=True)
u_array = np.load("u_pt_array_1.npy", allow_pickle=True)
v_array = np.load("v_pt_array_1.npy", allow_pickle=True)

eta_tensor = torch.tensor(eta_array, dtype=torch.float64, device=device)
# 对 u, v 做 pad 以满足尺寸要求
u_array = np.pad(u_array, ((0, 0), (0, 1), (0, 0)), mode='edge')
v_array = np.pad(v_array, ((0, 0), (0, 0), (0, 1)), mode='edge')

Fx_array = np.zeros((v_array.shape[0], Nx, Ny))
Fy_array = np.zeros((v_array.shape[0], Nx, Ny))

# 构造输入 X（包括 eta, u, v 三个通道）和输出 Y（这里虽然构造了，但训练中不使用 Y，loss 直接与外部 eta 比较）
X = np.stack([eta_array, u_array, v_array], axis=1)  
Y = np.stack([Fx_array, Fy_array], axis=1)            

num_total_steps = X.shape[0]
num_time_steps_train = 160     
num_time_steps_val = 40

X_tensor = torch.tensor(X, dtype=torch.float64)
Y_tensor = torch.tensor(Y, dtype=torch.float64)


input_channel = 3
output_channel = 2
model = CNN1(shapeX=input_channel).to(device)
model.double()  
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 使物理模型状态参与梯度计算
H = H_init.clone().detach().requires_grad_(True).to(device)
M = M_init.clone().detach().requires_grad_(True).to(device)
N = N_init.clone().detach().requires_grad_(True).to(device)
torch.autograd.set_detect_anomaly(True)

# 优化后的训练循环：直接利用整个训练序列（X_tensor）进行多步模拟
num_epochs = 40
chunk_size = 10   # 每隔多少步反传一次

train_loss_history = []
test_loss_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0  # 累计当前 epoch 的总 loss
    acc_loss = 0.0      # 累计 chunk 内 loss

    # 重新初始化物理状态，保证每个 epoch 从初始状态开始
    H = H_init.clone().detach().requires_grad_(True).to(device)
    M = M_init.clone().detach().requires_grad_(True).to(device)
    N = N_init.clone().detach().requires_grad_(True).to(device)

    for t in range(num_time_steps_train):
        # 当前输入：取第 t 个时间步（shape: [1, 3, Nx, Ny]）
        current_input = X_tensor[t].unsqueeze(0).to(device)
        output = model(current_input).squeeze(0)  # 输出 shape: [2, Nx, Ny]
        Fx_t, Fy_t = output[0], output[1]
        Fx_t_mean = Fx_t.mean().item()
        Fy_t_mean = Fy_t.mean().item()

        # 自适应 CFL 计算当前 dt
        dt_cfl = compute_dt_cfl(H, M, N, Z, dx, dy, g, cfl_factor)
        dt_current = min(dt, dt_cfl)

        # 物理更新
        H_new = mass_cartesian_torch(H, Z, M, N, dt_current)
        D_M, D_N = reconstruct_flow_depth_torch(H_new, Z, M, N, dry_limit)
        M_new, N_new = momentum_nonlinear_cartesian(H_new, Z, M, N, D_M, D_N, dt_current, Fx_t, Fy_t)
        H_new, M_new, N_new = apply_closed_boundary(H_new, M_new, N_new, Z)
        
        print("eta_tensor[t] min =", eta_tensor[t].min().item())
        # 计算当前时间步的 loss，与外部 eta_tensor 比较（保持不变）
        loss_t = criterion(H_new[1], eta_tensor[t])
        acc_loss += loss_t

        # 分段反向传播：每 chunk_size 步更新一次
        if (t + 1) % chunk_size == 0 or (t == num_time_steps_train - 1):
            optimizer.zero_grad()
            acc_loss.backward()
            optimizer.step()

            running_loss += acc_loss.item()
            # 截断计算图，更新状态
            H = H_new.detach().requires_grad_(True)
            M = M_new.detach().requires_grad_(True)
            N = N_new.detach().requires_grad_(True)
            acc_loss = 0.0
        else:
            # 继续传播梯度，不 detach
            H, M, N = H_new, M_new, N_new
            print("M_new[1] min =", M_new[1].min().item())
            print("N_new[1] min =", N_new[1].min().item())
            
            print("H_new[1] min =", H_new[1].min().item())
            print("H_new[1] max =", H_new[1].max().item())
            
            print("H_new[0] min =", H_new[0].min().item())
            print("H_new[0] max =", H_new[0].max().item())
            
            print("H[0] min =", H[0].min().item())
            print("H[0] max =", H[0].max().item())
            
            print("H[1] min =", H[1].min().item())
            print("H[1] max =", H[1].max().item())

    avg_train_loss = running_loss / num_time_steps_train
    train_loss_history.append(avg_train_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")

    model.eval()
    test_running_loss = 0.0
    with torch.no_grad():

        H_val = H_init.clone().to(device)
        M_val = M_init.clone().to(device)
        N_val = N_init.clone().to(device)
        for t in range(num_time_steps_val):
            current_input_val = X_tensor[t].unsqueeze(0).to(device)
            output_val = model(current_input_val).squeeze(0)
            Fx_val, Fy_val = output_val[0], output_val[1]
            dt_cfl_val = compute_dt_cfl(H_val, M_val, N_val, Z, dx, dy, g, cfl_factor)
            dt_current_val = min(dt, dt_cfl_val)
            H_new_val = mass_cartesian_torch(H_val, Z, M_val, N_val, dt_current_val)
            D_M_val, D_N_val = reconstruct_flow_depth_torch(H_new_val, Z, M_val, N_val, dry_limit)
            M_new_val, N_new_val = momentum_nonlinear_cartesian(H_new_val, Z, M_val, N_val, D_M_val, D_N_val, dt_current_val, Fx_val, Fy_val)
            H_new_val, M_new_val, N_new_val = apply_closed_boundary(H_new_val, M_new_val, N_new_val, Z)
            loss_val = criterion(H_new_val[1], eta_tensor[num_time_steps_val + t])
            test_running_loss += loss_val.item()
            H_val, M_val, N_val = H_new_val, M_new_val, N_new_val

        avg_test_loss = test_running_loss / num_time_steps_val
        test_loss_history.append(avg_test_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {avg_test_loss:.6f}")

