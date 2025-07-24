# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 19:48:23 2025

@author: ASUS
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
from matplotlib.pyplot import pcolor,colorbar


#eps = 1.0e-6

def tic():
    global _start_time
    _start_time = time.time()

def toc():
    elapsed_time = time.time() - _start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
def dd(var): #detach and operations
    return var.detach().cpu().numpy().squeeze()

def ddx_old(Y, scheme='edge'):
    #1st order difference along first dimension
    # - Y: 2D tensor
    # - scheme: 'inner' no padding, 'edge' pad so that dimension unchanged
    if scheme=='edge':
        pad = 1
    elif scheme=='inner':
        pad = 0
    # torch.roll
    kernel = torch.tensor([-1,1], dtype=Y.dtype,device=Y.device).unsqueeze(0).unsqueeze(0) 
    
    dMdx = torch.stack([
        F.conv1d(Y[:, i].unsqueeze(0).unsqueeze(0), kernel, padding=pad)
        for i in range(Y.size(1)) ], dim=2).squeeze().t()
        
    return dMdx

def ddx(Y, scheme='edge'):
    """沿第一维度（行）计算一阶差分
    输入形状: [L, M]
    输出形状:
      - scheme='edge' : [L, M]（边界填充保持维度）
      - scheme='inner': [L-1, M]（无填充）
    """
    # 确定填充参数
    padding = 1 if scheme == 'edge' else 0

    # 调整输入形状为 [batch=1, channels=M, length=L]
    Y_reshaped = Y.unsqueeze(0).permute(0, 2, 1)  # [1, M, L]

    # 创建差分核 [-1, 1]，形状为 [M, 1, 2]
    kernel = torch.tensor([-1.0, 1.0], dtype=Y.dtype, device=Y.device)
    kernel = kernel.view(1, 1, -1).repeat(Y.size(1), 1, 1)  # [M, 1, 2]

    # 应用分组卷积（每个列独立处理）
    dMdx = F.conv1d(Y_reshaped, kernel, padding=padding, groups=Y.size(1))  # [1, M, L] 或 [1, M, L+1]

    # 调整形状并转置
    return dMdx.permute(0, 2, 1).squeeze(0) 
    
def ddy_old(Y, scheme='edge'):
    #1st order difference along second dimension
    # - Y: 2D tensor
    # - scheme: 'inner' no padding, 'edge' pad so that dimension unchanged
    if scheme=='edge':
        pad = 1
    elif scheme=='inner':
        pad = 0
    kernel = torch.tensor([-1,1], dtype=Y.dtype,device=Y.device).unsqueeze(0).unsqueeze(0) #[1,1,2]
    dNdy = torch.stack([ 
        F.conv1d(Y[i, :].unsqueeze(0).unsqueeze(0), kernel, padding=pad)
        for i in range(Y.size(0)) ], dim=0).squeeze() #no t()
    return dNdy

def ddy(Y, scheme='edge'):
    """沿第二维度（列）计算一阶差分
    输入形状: [L, M]
    输出形状:
      - scheme='edge' : [L, M]（边界填充保持维度）
      - scheme='inner': [L, M-1]（无填充）
    """
    # 确定填充参数
    padding = 1 if scheme == 'edge' else 0

    # 调整输入形状为 [batch=L, channels=1, length=M]
    Y_reshaped = Y.unsqueeze(1)  # [L, 1, M]

    # 创建差分核 [-1, 1]，形状为 [1, 1, 2]
    kernel = torch.tensor([-1.0, 1.0], dtype=Y.dtype, device=Y.device)
    kernel = kernel.view(1, 1, -1)  # [1, 1, 2]

    # 批量卷积操作（一次性处理所有行）
    dNdy = F.conv1d(Y_reshaped, kernel, padding=padding)  # 输出形状 [L, 1, M] 或 [L, 1, M-1]

    # 去除通道维度
    return dNdy.squeeze(1)

def rho2u_old(Y):
    #from rho-point to u-point (-1 at first dimension)
    # - Y: 2D tensor
    #start = time.time()
    kernel = torch.tensor([.5,.5], dtype=Y.dtype,device=Y.device).unsqueeze(0).unsqueeze(0) #[1,1,2]
    H_u = torch.stack([
        F.conv1d(Y[:, i].unsqueeze(0).unsqueeze(0), kernel, padding=0)
        for i in range(Y.size(1)) ], dim=2).squeeze().t()
    #print(f'{time.time() - start}')
    return H_u

def rho2u(Y):
    # 输入 Y 形状为 [length=32, num_columns=39]
    # 调整形状为 [batch=1, channels=39, length=32]
    Y_reshaped = Y.t().unsqueeze(0)  # [1, 39, 32]
    # 创建卷积核：每个通道独立使用相同的核 [0.5, 0.5]
    kernel = torch.tensor([0.5, 0.5], dtype=Y.dtype, device=Y.device)
    kernel = kernel.repeat(Y.size(1), 1, 1)  # 形状 [39, 1, 2]
    # 应用分组卷积（每个通道独立处理）
    H_u = F.conv1d(Y_reshaped, kernel, groups=Y.size(1))  # 输出 [1, 39, 31]
    # 调整形状为 [31, 39]
    H_u = H_u.squeeze(0).t()  # 先压缩批次维度，再转置
    return H_u


def rho2v_old(Y):
    #from rho-point to v-point (-1 at second dimension)
    # - Y: 2D tensor
    kernel = torch.tensor([.5,.5], dtype=Y.dtype,device=Y.device).unsqueeze(0).unsqueeze(0) #[1,1,2]
    H_v = torch.stack([ 
        F.conv1d(Y[i, :].unsqueeze(0).unsqueeze(0), kernel, padding=0)
        for i in range(Y.size(0)) ], dim=0).squeeze() #no t()
    return H_v

def rho2v(Y):
    Y_reshaped = Y.unsqueeze(1)  # [L, 1, M]
    kernel = torch.tensor([0.5, 0.5], dtype=Y.dtype,device=Y.device).view(1, 1, 2)  # 核形状 [1, 1, 2]
    H_v = F.conv1d(Y_reshaped, kernel, padding=0).squeeze(1)
    return H_v

def u2rho_old(Y, mode ='default'):
    #from u-point to rho-point (-1 at first dimension)
    # - Y: 2D tensor
    if mode == 'expand':
        Y = F.pad(Y,[0,0,1,1])
    kernel = torch.tensor([.5,.5], dtype=Y.dtype,device=Y.device).unsqueeze(0).unsqueeze(0) #[1,1,2]
    Yr = torch.stack([
        F.conv1d(Y[:, i].unsqueeze(0).unsqueeze(0), kernel, padding=0)
        for i in range(Y.size(1)) ], dim=2).squeeze().t()
    return Yr

def u2rho(Y, mode='default'):
    """将u点转换为rho点（行维度减少1）
    输入形状: [L, M]
    输出形状: 
      - mode='default': [L-1, M] 
      - mode='expand'  : [L+1, M]
    """
    # 模式扩展处理：在行维度上下各填充1
    if mode == 'expand':
        Y = F.pad(Y, [0, 0, 1, 1])  # [left, right, top, bottom] -> 输出形状 [L+2, M]

    # 调整输入形状为 [batch=1, channels=M, length=L] 或 [1, M, L+2]
    Y_reshaped = Y.t().unsqueeze(0)  # [1, M, L] 或 [1, M, L+2]

    # 创建卷积核：每个通道独立处理，核形状 [M, 1, 2]
    kernel = torch.tensor([0.5, 0.5], dtype=Y.dtype, device=Y.device)
    kernel = kernel.repeat(Y.size(1), 1, 1)  # 形状 [M, 1, 2]

    # 应用分组卷积（每个列独立处理）
    Yr = F.conv1d(Y_reshaped, kernel, groups=Y.size(1))  # 输出形状 [1, M, L-1] 或 [1, M, L+1]

    # 调整形状并转置
    return Yr.squeeze(0).t()  # 最终形状 [L-1, M] 或 [L+1, M]

def v2rho_old(Y, mode ='default'):
    #from v-point to rho-point (-1 at second dimension)
    # - Y: 2D tensor
    if mode == 'expand':
        Y = F.pad(Y,[1,1,0,0])
    kernel = torch.tensor([.5,.5], dtype=Y.dtype,device=Y.device).unsqueeze(0).unsqueeze(0) #[1,1,2]
    Yr = torch.stack([ 
        F.conv1d(Y[i, :].unsqueeze(0).unsqueeze(0), kernel, padding=0)
        for i in range(Y.size(0)) ], dim=0).squeeze() #no t()
    return Yr

def v2rho(Y, mode='default'):
    # 输入形状: [L, M] (例如 [32, 39])
    if mode == 'expand':
        # 在第二维度（列）左右各填充1，格式为 [left, right, top, bottom]
        Y = F.pad(Y, [1, 1, 0, 0])  # 输出形状变为 [L, M+2]
    
    # 调整形状为 [L, 1, M] 或 [L, 1, M+2]
    Y_reshaped = Y.unsqueeze(1)  # [L, 1, M] 或 [L, 1, M+2]    
    # 创建卷积核 [0.5, 0.5]，形状为 [1, 1, 2]
    kernel = torch.tensor([0.5, 0.5], dtype=Y.dtype, device=Y.device)
    kernel = kernel.view(1, 1, -1)  # [1, 1, 2]    
    # 批量卷积操作（一次性处理所有行）
    Yr = F.conv1d(Y_reshaped, kernel, padding=0)  # 输出形状 [L, 1, M-1] 或 [L, 1, M+1]
    
    # 去除通道维度
    return Yr.squeeze(1)

# def rho2u(Y, scheme='inner'):
#     #from rho-point to u-point (-1 at first dimension)
#     # - Y: 2D tensor
#     # - scheme: 'inner' no padding, 'edge' pad so that dimension unchanged
#     if scheme=='inner':
#         out = (torch.roll(Y,1,0) + Y)[1:]/2
#     elif scheme=='edge':
#         out = (torch.roll(Y,1,0) + Y)/2
#     return out

# def rho2v(Y):
#     #from rho-point to u-point (-1 at first dimension)
#     # - Y: 2D tensor
#     #start = time.time()
#     #print(f'{time.time() - start :.9f}')
#     return (torch.roll(Y,1,1) + Y)[:,1:]/2

# def u2rho(Y): #equivalent to rho2u
#     #from u-point to rho-point (-1 at first dimension)
#     # - Y: 2D tensor
#     return (torch.roll(Y,1,0) + Y)[1:]/2

# def v2rho(Y): #equivalent to rho2v
#     #from v-point to rho-point (-1 at second dimension)
#     # - Y: 2D tensor
#     return (torch.roll(Y,1,1) + Y)[:,1:]/2

def ududx_up(M,N,H, eps=1e-12): #(16)
    # Upwind advection, x direction
    # only interior points were calculated, M [1:-1]
    #[1 ~ Nx  , 1 ~ Ny  ] for rho
    #[1 ~ Nx-2, 1 ~ Ny-1] for u 
    #[1 ~ Nx-1, 1 ~ Ny-2] for v
    # M = torch.randn([100,51])
    # N = torch.randn([101,50])
    # H = torch.randn([101,51])
    Hu_ = rho2u(H) #H at u-point 1/2, 3/2, 5/2, 0 ~ Nx, total of Nx
    Hu = torch.where(Hu_==0, Hu_ + eps, Hu_)
    #Mr = u2rho(M) #M at rho-point 1,2,3        1 ~ Nx, total of Nx-1
    
    L12 = torch.sign(M + 1e-12) # 0 ~ Nx-1, total of Nx
    L11 = torch.floor((1 - L12) / 2)# 0 ~ Nx-1, total of Nx
    L13 = -L12 - L11            # 0 ~ Nx-1, total of Nx
    assert torch.all(L11+L12+L13 == 0)
    assert torch.all(L11 < 2)
    
    t1 = L11[1:-1] * M[2:  ]**2 /Hu[2:  ]   # Nx - 2
    t2 = L12[1:-1] * M[1:-1]**2 /Hu[1:-1]  # 
    t3 = L13[1:-1] * M[ :-2]**2 /Hu[ :-2]
    
    ududx = t1 + t2 + t3
           
    return ududx

def vdudy_up(M,N,H, eps=1e-12):
    # Upwind advection, x direction
    # only interior points were calculated, M [1:-1]
    #[1 ~ Nx  , 1 ~ Ny  ] for rho
    #[1 ~ Nx-2, 1 ~ Ny-1] for u 
    #[1 ~ Nx-1, 1 ~ Ny-2] for v
    # M = torch.randn([100,51])
    # N = torch.randn([101,50])
    # H = torch.randn([101,51])
    N_exp = torch.cat( (N[:,0:1], N, N[:,-1:]), dim = 1)
    Nu = rho2u(v2rho(N_exp)) #at u-point, [ 0 ~ Nx, 1 ~ Ny-1]
    
    #These 
    L22 = torch.sign(Nu + 1e-12) # same with Nu
    L21 = torch.floor((1 - L22) / 2)
    L23 = -L22 - L21             #
    assert torch.all(L21+L22+L23 == 0) and torch.all(L21 < 2)
    
    Hu_ = rho2u(H)  #H at u-point 1/2, 3/2, 5/2, 0 ~ Nx, total of Nx
    Hu = torch.where(Hu_==0, Hu_ + eps, Hu_)
    
    # M all at i + 1/2 
    t1 = L21[:,1:-1] * M[:, 2:  ] * Nu[:,2:  ] / Hu[:,2:  ]   # [Nx - 1, Ny - 2]
    t2 = L22[:,1:-1] * M[:, 1:-1] * Nu[:,1:-1] / Hu[:,1:-1]   # 
    t3 = L23[:,1:-1] * M[:,  :-2] * Nu[:, :-2] / Hu[:, :-2]
    
    vdudy = t1 + t2 + t3
           
    return vdudy


def udvdx_up(M,N,H, eps=1e-12):
    # Upwind advection, y direction
    # only interior points were calculated, M [1:-1]
    #[1 ~ Nx  , 1 ~ Ny  ] for rho
    #[1 ~ Nx-2, 1 ~ Ny-1] for u 
    #[1 ~ Nx-1, 1 ~ Ny-2] for v
    # M = torch.randn([100,51])
    # N = torch.randn([101,50])
    # H = torch.randn([101,51])
    Hv_ = rho2v(H)       #H at v-point I, J+1/2, 3/2, 5/2, 0 ~ Ny, total of Ny
    Hv = torch.where(Hv_==0, Hv_ + eps, Hv_) 
    #Nr = v2rho(N)        #M at rho-point 1,2,3        1 ~ Nx, total of Nx-1
    M_exp = torch.cat( (M[0:1],M, M[-1:]), dim =0)
    Mv = rho2u(v2rho(M_exp)) #at v-point, [ 1 ~ Nx-1, 1 ~ Ny-1]
    
    L32 = torch.sign(Mv + 1e-12) # 0 ~ Nx-1, total of Nx
    L31 = torch.floor((1 - L32) / 2)# 0 ~ Nx-1, total of Nx
    L33 = -L32 - L31            # 0 ~ Nx-1, total of Nx
    assert torch.all(L31+L32+L33 == 0) and torch.all(L31 < 2)
    
    t1 = L31[1:-1] * Mv[2:  ] * N[2:  ]/Hv[2:  ]   # Nx-1, Ny
    t2 = L32[1:-1] * Mv[1:-1] * N[1:-1]/Hv[1:-1]  # Nx-1, Ny
    t3 = L33[1:-1] * Mv[ :-2] * N[ :-2]/Hv[ :-2]    # Nx-1, Ny
    
    udvdx = t1 + t2 + t3
           
    return udvdx


def vdvdy_up(M,N,H, eps=1e-12): #done
    # Upwind advection, y direction
    # only interior points were calculated, M [1:-1]
    #[1 ~ Nx  , 1 ~ Ny  ] for rho
    #[1 ~ Nx-2, 1 ~ Ny-1] for u 
    #[1 ~ Nx-1, 1 ~ Ny-2] for v
    # M = torch.randn([100,51])
    # N = torch.randn([101,50])
    # H = torch.randn([101,51])
    
    L42 = torch.sign(N + 1e-12)  # 
    L41 = torch.floor((1 - L42) / 2)
    L43 = -L42 - L41             #
    assert torch.all(L41+L42+L43 == 0) and torch.all(L41 < 2)
    
    Hv_ = rho2v(H) #H at v-point 0 ~ Ny, total of Ny
    Hv = torch.where(Hv_==0, Hv_ + eps, Hv_)
    # M all at i + 1/2
    t1 = L41[:,1:-1] * N[:, 2:  ] **2 / Hv[:,2:  ]   # [Nx - 1, Ny - 2]
    t2 = L42[:,1:-1] * N[:, 1:-1] **2 / Hv[:,1:-1]   # 
    t3 = L43[:,1:-1] * N[:, :-2]  **2 / Hv[:, :-2]
    
    vdvdy = t1 + t2 + t3
           
    return vdvdy
