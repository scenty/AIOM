import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#from CNN_noip import CNN1  
#from torch.utils.data import TensorDataset, DataLoader   # 不再使用 DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from matplotlib.pyplot import *

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
    friction_term_M = torch.where(friction_mask, dt * Fx[:Nx_m, :]/10000000, torch.zeros_like(Fx[:Nx_m, :]))
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
                                      dt * Fy[:, :Ny_n-1]/100000000,
                                      torch.zeros_like(Fy[:, :Ny_n-1]))
        N_val_updated = torch.cat((N_val[:, :-1] - friction_term_N, N_val[:, -1:]), dim=1)
    else:
        N_val_updated = N_val

    N_new = torch.stack((N[0], N_val_updated), dim=0)
    return M_new, N_new, Fx, Fy

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