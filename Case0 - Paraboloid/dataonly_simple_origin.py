# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 00:48:58 2025

@author: ASUS
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

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

x_t = torch.linspace(-Lx/2, Lx/2, Nx)  # double precision by default now
y_t = torch.linspace(-Ly/2, Ly/2, Ny)
X, Y = torch.meshgrid(x_t, y_t, indexing='ij')  # shape (Nx, Ny)

g = 9.81
manning = 0.02
dry_limit = 1e5
MinWaterDepth = 1e-4
FrictionDepthLimit = 5e-3
cfl_factor = 0.9  # 安全系数

# 先给个初始 dt
dt = 1
NT = 200  # 最大步数

# 一些系数
def c1(dt, dx): return dt/dx
def c2(dt, dy): return dt/dy

##############################################################################
# 2. 床面与初始水面
##############################################################################
# def bed_elevation_torch(X, Y, D0, L):
#     r = torch.sqrt(X**2 + Y**2)
#     # 令碗中心处床面为负
#     return -D0*(1.0 - r**2 / L**2)

# def stage_init_torch(X, Y, D0, L, A, omega, t):
#     z = bed_elevation_torch(X, Y, D0, L)
#     r = torch.sqrt(X**2 + Y**2)
#     factor = (1 - A**2) / ((1 - A*torch.cos(omega*t))**2) - 1.0
#     w = D0*( (torch.sqrt(1 - A**2))/(1 - A*torch.cos(omega*t)) - 1.0 ) \
#         - D0*(r**2 / L**2)*factor
#     # 保证水面不低于床面
#     w = torch.max(w, z)
#     return w

# import torch

def bed_elevation_torch(X, Y, D0, L):
    """
    对应您 NumPy 版本的 bed_elevation(x, y):
      z = -D0*(1. - r^2/(L^2))
    这里 X, Y, z 的形状均为 (Nx, Ny)。
    """
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


# 生成床面 & 初始水面
D0 = torch.tensor(1.0)
L = torch.tensor(2500.0)
R0 = torch.tensor(2000.0)
# 初始条件
t = 0.0
A = (L ** 4 - R0 ** 4) / (L ** 4 + R0 ** 4)
omega = 2. / L * torch.sqrt(2. * g * D0)

# 然后直接:
Z = bed_elevation_torch(X, Y, D0, L)
Z = torch.tensor(Z)
# H_init = stage_init_torch(X, Y, D0, L, A, omega, t=0.0)
H_init = stage_init_torch(X, Y, D0, L, A, omega, t=0.0)
plot(Z[35]);plot(H_init[35]);legend(('Z','H_init'))
#print("H_init[0] min =", H_init[0].min().item())
#print("H_init[0] max =", H_init[0].max().item())
##############################################################################
# 3. 分配张量：H, M, N
##############################################################################
# H: (2, Nx, Ny)
H = torch.zeros((2, Nx, Ny), dtype=torch.float64)
H[0] = H_init.clone()

# M: x方向动量，(2, Nx-1, Ny)
M = torch.zeros((2, Nx-1, Ny), dtype=torch.float64)
# N: y方向动量，(2, Nx, Ny-1)
N = torch.zeros((2, Nx, Ny-1), dtype=torch.float64)

##############################################################################
# 4. 辅助函数：check_flux_direction
##############################################################################
def check_flux_direction_torch(z1, z2, h1, h2, dry_limit):
    device = z1.device
    # 克隆 h1、h2，确保后续修改不会直接影响原始数据
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

    # 干湿修正（使用 torch.where 保证全为新张量）
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

    # 构造新的 H，不再对 H.clone() 进行切片赋值，而是用 stack 构造新张量
    H_new = torch.stack((H0, H1), dim=0)
    return H_new


##############################################################################
# 7. 动量方程 (momentum_nonlinear_cartesian)
#############################################################################
def momentum_nonlinear_cartesian(H, Z, M, N, D_M, D_N, dt):
    """
    与之前类似的上风 + 干湿 + 底摩擦。
    """
    Dlimit = 1e-3

    # ========== M 分量 (Nx-1, Ny) ==========
    m0 = M[0]
    Nx_m, Ny_m = m0.shape
    D0 = D_M
    # 构造 m1, m2
    m1 = torch.zeros_like(m0)
    m2 = torch.zeros_like(m0)
    if Nx_m > 1:
        m1[1:, :] = m0[:-1, :]
        m2[:-1, :] = m0[1:, :]

    D1 = torch.zeros_like(D0)
    D2 = torch.zeros_like(D0)
    if Nx_m > 1:
        D1[1:, :] = D0[:-1, :]
        D2[:-1, :] = D0[1:, :]

    z1_M = Z[:-1, :]
    z2_M = Z[1:, :]
    h1_M = H[1, :-1, :]
    h2_M = H[1, 1:, :]
    flux_sign_M, h1u_M, h2u_M = check_flux_direction_torch(z1_M, z2_M, h1_M, h2_M, dry_limit)

    cond_m0pos = (m0 >= 0)
    maskD0 = (D0 > Dlimit)
    maskD1 = (D1 > Dlimit)
    maskD2 = (D2 > Dlimit)

    MU1 = torch.zeros_like(m0)
    MU2 = torch.zeros_like(m0)
    MU1 = torch.where(cond_m0pos & maskD1, m1**2 / D1, MU1)
    MU2 = torch.where(cond_m0pos & maskD0,  m0**2 / D0, MU2)
    MU1 = torch.where(~cond_m0pos & maskD0, m0**2 / D0, MU1)
    MU2 = torch.where(~cond_m0pos & maskD2, m2**2 / D2, MU2)

    phi_M = 1.0 - (dt/dx)*torch.min(torch.abs(m0 / torch.clamp(D0, min=MinWaterDepth)), torch.sqrt(g*D0))
    M_val = phi_M*m0 + 0.5*(1.0 - phi_M)*(m1 + m2) \
            - (dt/dx)* (MU2 - MU1) \
            - (dt*g/dx)* D0*(h2u_M - h1u_M)

    mask_fs999_M = (flux_sign_M == 999)
    M_val = torch.where(mask_fs999_M, torch.zeros_like(M_val), M_val)

    # 底摩擦
    friction_mask = (D0 > FrictionDepthLimit)
    Cf = manning
    MM = m0**2
    NN = torch.zeros_like(m0)
    Fx = g * Cf**2 / (D0**2.33) * torch.sqrt(MM + NN) * m0
    Fx = torch.where(torch.abs(dt*Fx)>torch.abs(m0), m0/dt, Fx)
    M_val = M_val - torch.where(friction_mask, dt*Fx, torch.zeros_like(Fx))

    M_new = M.clone()
    M_new[1] = M_val

    # ========== N 分量 (Nx, Ny-1) ==========
    n0 = N[0]
    Nx_n, Ny_n = n0.shape
    D0N = D_N

    n1 = torch.zeros_like(n0)
    n2 = torch.zeros_like(n0)
    if Ny_n > 1:
        n1[:, 1:] = n0[:, :-1]
        n2[:, :-1] = n0[:, 1:]

    D1N = torch.zeros_like(D0N)
    D2N = torch.zeros_like(D0N)
    if Ny_n > 1:
        D1N[:, 1:] = D0N[:, :-1]
        D2N[:, :-1] = D0N[:, 1:]

    z1_N = Z[:, :-1]
    z2_N = Z[:, 1:]
    h1_N = H[1, :, :-1]
    h2_N = H[1, :, 1:]
    flux_sign_N, h1u_N, h2u_N = check_flux_direction_torch(z1_N, z2_N, h1_N, h2_N, dry_limit)

    cond_n0pos = (n0 >= 0)
    maskD0N = (D0N > Dlimit)
    maskD1N = (D1N > Dlimit)
    maskD2N = (D2N > Dlimit)

    NV1 = torch.zeros_like(n0)
    NV2 = torch.zeros_like(n0)
    NV1 = torch.where(cond_n0pos & maskD1N, n1**2 / D1N, NV1)
    NV2 = torch.where(cond_n0pos & maskD0N, n0**2 / D0N, NV2)
    NV1 = torch.where(~cond_n0pos & maskD0N, n0**2 / D0N, NV1)
    NV2 = torch.where(~cond_n0pos & maskD2N, n2**2 / D2N, NV2)

    phi_N = 1.0 - (dt/dy)*torch.min(torch.abs(n0 / torch.clamp(D0N, min=MinWaterDepth)), torch.sqrt(g*D0N))
    N_val = phi_N*n0 + 0.5*(1.0 - phi_N)*(n1 + n2) \
            - (dt/dy)*(NV2 - NV1) \
            - (dt*g/dy)* D0N*(h2u_N - h1u_N)

    mask_fs999_N = (flux_sign_N == 999)
    N_val = torch.where(mask_fs999_N, torch.zeros_like(N_val), N_val)

    friction_maskN = (D0N > FrictionDepthLimit)
    MM_N = torch.zeros_like(n0)
    NN_N = n0**2
    Fy = g * Cf**2 / (D0N**2.33) * torch.sqrt(MM_N + NN_N) * n0
    Fy = torch.where(torch.abs(dt*Fy)>torch.abs(n0), n0/dt, Fy)
    N_val = N_val - torch.where(friction_maskN, dt*Fy, torch.zeros_like(Fy))

    N_new = N.clone()
    N_new[1] = N_val

    return M_new, N_new

##############################################################################
# 8. 自适应 CFL：计算最大波速 -> 调整 dt
##############################################################################
def compute_dt_cfl(H, M, N, Z, dx, dy, g, cfl_factor=0.9):
    # 保证 Z 是 double 且在 device 上
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
# 9. 辅助：边界处理(封闭域) & 钳制水深
##############################################################################
# def apply_closed_boundary(H, M, N, Z):
#     """
#     同时实现：
#     1) 碗外(或任何 Z>0 的区域) 强制无水、动量=0
#     2) 最外一圈边界: 水面夹在 [床面, 0], 动量=0
#     """

#     # ========== 1) 整个网格里，若 Z>0 则 H=0, 动量=0 ==========
#     Nx_, Ny_ = Z.shape
#     # H: (2, Nx_, Ny_), M: (2, Nx_-1, Ny_), N: (2, Nx_, Ny_-1)

#     # 这里以 layer=1(新层) 为例，也可以对 [0,1] 都做
#     layer = 1  # 假设只对最新的那层做处理
#     # 第一步：找出 H[1] < 0 的位置
#     threshold = 0.01
#     stage = Z - H[layer]  # stage = 床面 + 水深

#     # 找出那些 stage 的绝对值在阈值内的区域
#     mask_bedpos = (torch.abs(stage) < threshold)

#     # 检查全域(或接触面区域)内，是否存在任何 stage<0 的点
#     any_neg = (H[layer][mask_bedpos] < 0.05).any()
    
#     if any_neg:
#         # 若发现有些点 <0，则将所有 >0 的点统一强制为0
#         mask_pos = (stage < 0)
#         H[layer][mask_pos] = 0.0
    
#         # 对应 M, N 需要找“横向/纵向索引”都在碗外的格子才置0。
#         # M 对应 x方向通量 (Nx_-1, Ny_)，可用 mask_bedpos[:-1,:] & mask_bedpos[1:,:] 判断左右两个网格点是否都在干区
#         if Nx_ > 1:
#             mask_M = mask_bedpos[:-1,:] & mask_bedpos[1:,:]
#             M[layer][mask_M] = 0.0
#         # N 对应 y方向通量 (Nx_, Ny_-1)，可用 mask_bedpos[:, :-1] & mask_bedpos[:, 1:] 判断上下两个网格点是否都在干区
#         if Ny_ > 1:
#             mask_N = mask_bedpos[:, :-1] & mask_bedpos[:, 1:]
#             N[layer][mask_N] = 0.0
    
#         # ========== 2) 对网格边界进行“封闭”处理 ==========
#         for layer_idx in [0, 1]:  # 也可以只对最新层
#             # -- 左右边界 --
#             # i=0
#             stage_left = Z[0, :] + H[layer_idx, 0, :]
#             # 夹在 [Z(0,:), 0] 区间
#             stage_left = torch.clamp(stage_left, min=Z[0, :], max=torch.tensor(0.0))
#             H[layer_idx, 0, :] = stage_left - Z[0, :]
    
#             # i = Nx_-1
#             stage_right = Z[-1, :] + H[layer_idx, -1, :]
#             stage_right = torch.clamp(stage_right, min=Z[-1, :], max=torch.tensor(0.0))
#             H[layer_idx, -1, :] = stage_right - Z[-1, :]
    
#             # -- 上下边界 --
#             # j=0
#             stage_bottom = Z[:, 0] + H[layer_idx, :, 0]
#             stage_bottom = torch.clamp(stage_bottom, min=Z[:, 0], max=torch.tensor(0.0))
#             H[layer_idx, :, 0] = stage_bottom - Z[:, 0]
    
#             # j=Ny_-1
#             stage_top = Z[:, -1] + H[layer_idx, :, -1]
#             stage_top = torch.clamp(stage_top, min=Z[:, -1], max=torch.tensor(0.0))
#             H[layer_idx, :, -1] = stage_top - Z[:, -1]
    
#             # 动量=0 (无通量)
#             if Nx_ > 1:
#                 M[layer_idx, 0, :]  = 0.0   # 左边界
#                 M[layer_idx, -1, :] = 0.0   # 右边界
#             if Ny_ > 1:
#                 N[layer_idx, :, 0]  = 0.0   # 下边界
#                 N[layer_idx, :, -1] = 0.0   # 上边界

#     return H, M, N

def apply_closed_boundary(H, M, N, Z):
    # 只对 layer=1 进行更新，其他层保持不变
    Nx_, Ny_ = Z.shape
    layer = 1

    # 提取需要更新的层，使用 clone 得到新变量
    H_layer = H[layer].clone()
    threshold = 0.005
    stage = Z - H_layer
    mask_bedpos = (torch.abs(stage) < threshold)

    # 如果床面附近存在较小的水深，则更新 H_layer
    if (H_layer[mask_bedpos] < 0.05).any():
        # 根据 stage 的正负调整 H_layer（这里使用 out-of-place 版本 torch.where）
        H_layer = torch.where(stage < 0, torch.zeros_like(H_layer), H_layer)

        # 计算各边界的更新值（各自为 1D 张量）
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

        # 使用行、列索引张量构造边界 mask，并用 torch.where 完成更新（全部 out-of-place）
        Nx = Nx_
        Ny = Ny_
        rows = torch.arange(Nx, device=H_layer.device).view(Nx, 1).expand(Nx, Ny)
        cols = torch.arange(Ny, device=H_layer.device).view(1, Ny).expand(Nx, Ny)

        # 对于第一行、最后一行、第一列和最后一列分别更新
        H_layer_updated = H_layer
        H_layer_updated = torch.where(rows == 0, left_boundary.unsqueeze(0).expand(Nx, Ny), H_layer_updated)
        H_layer_updated = torch.where(rows == Nx - 1, right_boundary.unsqueeze(0).expand(Nx, Ny), H_layer_updated)
        H_layer_updated = torch.where(cols == 0, bottom_boundary.unsqueeze(1).expand(Nx, Ny), H_layer_updated)
        H_layer_updated = torch.where(cols == Ny - 1, top_boundary.unsqueeze(1).expand(Nx, Ny), H_layer_updated)
    else:
        H_layer_updated = H_layer

    # 对 M 和 N 也采用完全 out-of-place 的方式处理边界
    # M_layer: shape 为 [Nx-1, Ny]
    M_layer = M[layer].clone()
    M_Nx, M_Ny = M_layer.shape
    M_rows = torch.arange(M_Nx, device=M_layer.device).view(M_Nx, 1).expand(M_Nx, M_Ny)
    M_layer_updated = M_layer
    if M_Nx > 1:
        M_layer_updated = torch.where(M_rows == 0, torch.zeros_like(M_layer_updated), M_layer_updated)
        M_layer_updated = torch.where(M_rows == M_Nx - 1, torch.zeros_like(M_layer_updated), M_layer_updated)

    # N_layer: shape 为 [Nx, Ny-1]
    N_layer = N[layer].clone()
    N_Nx, N_Ny = N_layer.shape
    N_cols = torch.arange(N_Ny, device=N_layer.device).view(1, N_Ny).expand(N_Nx, N_Ny)
    N_layer_updated = N_layer
    if N_Ny > 1:
        N_layer_updated = torch.where(N_cols == 0, torch.zeros_like(N_layer_updated), N_layer_updated)
        N_layer_updated = torch.where(N_cols == N_Ny - 1, torch.zeros_like(N_layer_updated), N_layer_updated)

    # 将更新后的状态与原始状态拼接（使用 torch.stack，不会引入 in-place 修改）
    H_new = torch.stack((H[0], H_layer_updated), dim=0)
    M_new = torch.stack((M[0], M_layer_updated), dim=0)
    N_new = torch.stack((N[0], N_layer_updated), dim=0)
    return H_new, M_new, N_new

##############################################################################
# 10. 主时间步进循环
##############################################################################
eta_list = []
u_list = []
v_list = []
dt_list = []

for it in range(NT):
    # 自适应CFL
    dt_cfl = compute_dt_cfl(H, M, N, Z, dx, dy, g, cfl_factor)
    dt = min(dt, dt_cfl)  # 也可直接 dt = dt_cfl

    # 更新水面
    H = mass_cartesian_torch(H, Z, M, N, dt)
    # 重构流深
    D_M, D_N = reconstruct_flow_depth_torch(H, Z, M, N, dry_limit)
    # 更新动量
    M, N = momentum_nonlinear_cartesian(H, Z, M, N, D_M, D_N, dt)

    # 同步
    H[0] = H[1].clone()
    M[0] = M[1].clone()
    N[0] = N[1].clone()

    # 额外操作：封闭边界 & 水深钳制

    H, M, N = apply_closed_boundary(H, M, N, Z)
    
    # 记录
    eta_list.append(H[1].cpu().numpy())
    u_list.append(M[1, :].cpu().numpy())
    v_list.append(N[1, :].cpu().numpy())
    dt_list.append(dt)

    # 也可动态显示 dt
    # print(f"Step {it}, dt={dt}, dt_cfl={dt_cfl}")

##############################################################################
# 11. 绘图
##############################################################################
eta_final = eta_list[-1]
plt.figure(figsize=(6,5))
plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), eta_final, 50, cmap='jet')
plt.colorbar(label='Stage')
plt.title("Final Water Surface (Stage)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()



# 保存为 NumPy 文件
np.save("eta_pt_array_1.npy", eta_list, allow_pickle=True)
np.save("u_pt_array_1.npy", u_list, allow_pickle=True)
np.save("v_pt_array_1.npy", v_list, allow_pickle=True)


M1 = np.array(M)