import numpy as np
#import vis_tools
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from J65 import *
from matplotlib.pyplot import *
import xarray as xr
import torch
import torch.nn.functional as F
from tools import ddx,ddy,rho2u,rho2v,dd
from tools import ududx_up,vdudy_up,udvdx_up,vdvdy_up

class Params:
    def __init__(self):
        # Domain parameters
        self.Lx = 80000
        self.Ly = 40000
        self.Nx = 100
        self.Ny = 50
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.depth = 50.0
        
        # Physical constants
        self.g = 9.8
        self.rho_water = 1025.0
        self.rho_air = 1.2
        self.Cd = 2.5e-3
        self.manning = 0.0
        self.dry_limit = 20
        self.MinWaterDepth = 0.01
        self.FrictionDepthLimit = 5e-3
        self.f_cor = 1e-5
        
        # Time parameters
        self.dt = 10
        self.NT = 1000
        self.centerWeighting0 = 0.9998
        
        # Boundary conditions
        self.obc_ele = ['Gra', 'Gra', 'Gra', 'Gra']
        self.obc_u2d = ['Gra', 'Gra', 'Gra', 'Gra'] 
        self.obc_v2d = ['Gra', 'Gra', 'Gra', 'Gra']
        
        # Temporary variables
        self.CC1 = self.dt / self.dx
        self.CC2 = self.dt / self.dy
        self.CC3 = self.CC1 * self.g
        self.CC4 = self.CC2 * self.g
        
        # Wind parameters
        self.Wr = 30
        self.rMax = 50*1000
        self.typhoon_Vec = [5,0]



def mass_cartesian_torch(H, Z, M, N, params):
    """
    params: structure containing all parameters including:
        - rho_air: air density (default 1.2)
        - rho_water: water density (default 1025.0)
        - Dlimit: depth limit (default 1.0e-3)
        - Cd: drag coefficient (default 2.5e-3)
        - f_cor: Coriolis parameter (default 1e-5)
        - CC1, CC2, CC3, CC4: coefficients
        - g: gravitational acceleration
        - MinWaterDepth: minimum water depth
        - FrictionDepthLimit: friction depth limit
        - manning: Manning coefficient
        - dt: time step
        - itime: current time step
        - H_update: updated water elevation
    """
    CC1 = params.CC1
    CC2 = params.CC2
    dry_limit = params.dry_limit
    MinWaterDepth = params.MinWaterDepth
        
    H0 = H[0]
    M0 = M[0]
    N0 = N[0]
    Nx, Ny = H0.shape
      
    dMdx = ddx(M0)
    dNdy = ddy(N0)
    
    H1 = H0.clone()
    H1[1:-1,1:-1] = H0[1:-1,1:-1] - CC1 * dMdx[1:-1,1:-1] - CC2 * dNdy[1:-1,1:-1]
        
    #TODO    
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
    H_new = bcond_zeta(H_new, Z, params)
    
    assert not torch.any(torch.isnan(H_new))
    #TODO boundary tide
    #H_new[1,1,:] = 1.0 * np.sin(2 * np.pi / 200 * params.itime * params.dt)
    return H_new

def momentum_nonlinear_cartesian_torch(H, Z, M, N, Wx, Wy, Pa, params):
    """
    - H: water elevation
    - Z: domain depth
    - M: x-direction momentum
    - N: y-direction momentum
    - Wx: wind u velocity in n+1 step, in rho point
    - Wy: wind v velocity in n+1 step, in rho point
    - Pa: Pressure in n+1 step, in rho point
    params: structure containing all parameters including:
        - rho_air: air density (default 1.2)
        - rho_water: water density (default 1025.0)
        - Dlimit: depth limit (default 1.0e-3)
        - Cd: drag coefficient (default 2.5e-3)
        - f_cor: Coriolis parameter (default 1e-5)
        - CC1, CC2, CC3, CC4: coefficients
        - g: gravitational acceleration
        - MinWaterDepth: minimum water depth
        - FrictionDepthLimit: friction depth limit
        - manning: Manning coefficient
        - dt: time step
        - itime: current time step
        - H_update: updated water elevation
    """
    # Extract parameters from params structure
    rho_air = getattr(params, 'rho_air', 1.2)
    rho_water = getattr(params, 'rho_water', 1025.0)
    Dlimit = getattr(params, 'Dlimit', 1.0e-3)
    Cd = getattr(params, 'Cd', 2.5e-3)
    f_cor = getattr(params, 'f_cor', 1e-5)
    CC1 = params.CC1
    CC2 = params.CC2
    CC3 = params.CC3
    CC4 = params.CC4
    g = params.g
    MinWaterDepth = params.MinWaterDepth
    FrictionDepthLimit = params.FrictionDepthLimit
    Cf = params.manning
    dt = params.dt
    phi = params.centerWeighting0
    Nx = params.Nx
    Ny = params.Ny
        
    # 重构流深, TODO(check Lechi的修改)
    D_M, D_N = reconstruct_flow_depth_torch(H, Z, M, N, params)
    # get forcing
    windSpeed = np.sqrt(Wx * Wx + Wy * Wy)
    sustr = rho2u( rho_air / rho_water * Cd * windSpeed * Wx)
    svstr = rho2v( rho_air / rho_water * Cd * windSpeed * Wy)
    
    # 以下是何乐驰版本
    # ========== M 分量 (Nx-1, Ny) ==========
    m0 = M[0]
    Nx_m, Ny_m = m0.shape
    D0 = D_M 
    # 构造 m1(i.e., m_(i-1) ), m2(mi+1)
    m1 = torch.roll(m0,-1, dims=0) #roll down, and discard last row
    m2 = torch.roll(m0, 1, dims=0) #roll up, and discard first row
    
    D1 = torch.roll(D0,-1, dims=0)
    D2 = torch.roll(D0, 1, dims=0)

    z1_M = Z[:-1, :]
    z2_M = Z[1:, :]
    h1_M = H[1, :-1, :]
    h2_M = H[1, 1:, :]
    flux_sign_M, h1u_M, h2u_M = check_flux_direction_torch(z1_M, z2_M, h1_M, h2_M, params.dry_limit)

    cond_m0pos = (m0 >= 0)
    maskD0 = (D0 > Dlimit)
    maskD1 = (D1 > Dlimit)
    maskD2 = (D2 > Dlimit)
    
    # Flux-centered, Lu started here
    dPdx = ddx(Pa,'inner')
    Pre_grad_x = CC1 * D0 * dPdx / rho_water
    
    ududx = F.pad( ududx_up(M[0],N[0],Z+H[1]), (0,0,1,1)) #pad for up and down
    vdudy = F.pad( vdudy_up(M[0],N[0],Z+H[1]), (1,1,0,0)) #pad for left and right
    
    Nu = F.pad( rho2u(rho2v(N[0])) , (1,1))
    # He&Lu Torch
    phi_M = 1.0 - CC1 * torch.min(torch.abs(m0 / torch.clamp(D0, min=MinWaterDepth)), torch.sqrt(g*D0))
    M_val = (phi_M*m0 + 0.5*(1.0 - phi_M)*(m1 + m2)        #implicit friction
            + dt  * ( sustr + f_cor * Nu)                  #wind stress and coriolis
            - CC1 * ududx                                  #u advection
            - CC2 * vdudy                                  #v advection
            - CC3 * D0*(h2u_M - h1u_M) + Pre_grad_x)         #pgf
    
    pcolor(rho2u(X),rho2u(Y),ududx);colorbar();show()
    pcolor(rho2u(X),rho2u(Y),vdudy);colorbar();show()
    
    mask_fs999_M = (flux_sign_M == 999)
    M_val = torch.where(mask_fs999_M, torch.zeros_like(M_val), M_val)

    # 底摩擦
    # friction_mask = (D0 > FrictionDepthLimit)
    # MM = m0**2
    # NN = torch.zeros_like(m0)
    # Fx = g * Cf**2 / (D0**2.33 + 1e-9) * torch.sqrt(MM + NN) * m0
    # Fx = torch.where(torch.abs(dt*Fx)>torch.abs(m0), m0/dt, Fx)
    # M_val = M_val - torch.where(friction_mask, dt*Fx, torch.zeros_like(Fx))

    M_new = M.clone()
    M_new[1,1:-1,1:-1] = M_val[1:-1,1:-1]

    # ========== N 分量 (Nx, Ny-1) ==========
    n0 = N[0]
    Nx_n, Ny_n = n0.shape
    D0N = D_N
    
    # 构造n1 (i.e., n_j+1), n2 (n_j-1) ##?????TODO check!
    n1 = torch.roll(n0, 1, dims=0) #roll right
    n2 = torch.roll(n0,-1, dims=0) #roll left
    
    D1N = torch.roll(D0N, 1, dims=0) #roll right
    D2N = torch.roll(D0N,-1, dims=0) #roll left
    
    z1_N = Z[:, :-1]
    z2_N = Z[:, 1:]
    h1_N = H[1, :, :-1]
    h2_N = H[1, :, 1:]
    flux_sign_N, h1u_N, h2u_N = check_flux_direction_torch(z1_N, z2_N, h1_N, h2_N, params.dry_limit)

    cond_n0pos = (n0 >= 0)
    maskD0N = (D0N > Dlimit)
    maskD1N = (D1N > Dlimit)
    maskD2N = (D2N > Dlimit)

    #Lu
    dPdy = ddy(Pa,'inner')
    Pre_grad_y = CC2 * D0N * dPdy / rho_water
    
    udvdx = F.pad( udvdx_up(M[0],N[0],Z+H[1]), (0,0,1,1)) #pad for up and down
    vdvdy = F.pad( vdvdy_up(M[0],N[0],Z+H[1]), (1,1,0,0)) #pad for left and right
    
    Mv = F.pad( rho2u(rho2v( M[0])) , (0,0,1,1)) #pad for up and down
    phi_N = 1.0 - CC2 * torch.min(torch.abs(n0 / torch.clamp(D0N, min=MinWaterDepth)), torch.sqrt(g*D0N))
    N_val = (phi_N*n0 + 0.5*(1.0 - phi_N)*(n1 + n2) 
            + dt * (svstr - f_cor * Mv )
            - CC1 * udvdx
            - CC2 * vdvdy
            - (CC4 * D0N * (h1u_N - h2u_N) + Pre_grad_y))
      
    mask_fs999_N = (flux_sign_N == 999)
    N_val = torch.where(mask_fs999_N, torch.zeros_like(N_val), N_val)

    # friction_maskN = (D0N > FrictionDepthLimit)
    # MM_N = torch.zeros_like(n0)
    # NN_N = n0**2
    # Fy = g * Cf**2 / (D0N**2.33 + 1e-9) * torch.sqrt(MM_N + NN_N) * n0
    # Fy = torch.where(torch.abs(dt*Fy)>torch.abs(n0), n0/dt, Fy)
    # N_val = N_val - torch.where(friction_maskN, dt*Fy, torch.zeros_like(Fy))

    N_new = N.clone()
    N_new[1,1:-1,1:-1] = N_val[1:-1,1:-1]
    pcolor(rho2v(X)[1:-1,1:-1],rho2v(Y)[1:-1,1:-1],N_val[1:-1,1:-1]/D_N[1:-1,1:-1]);colorbar();show()
    
    # apply boundary condition
    z_w = 0 #.05 * np.sin(2 * np.pi / 0.25 * itime * dt) * (np.zeros((Ny + 1, 1)) + 1)
    z_s = 0
    z_n = 0
    z_e = 0
    M_update = bcond_u2D(H_update, Z, M, D_M, z_w, z_e, params)
    N_update = bcond_v2D(H_update, Z, N, D_N, z_s, z_n, params)
    # M_new[:,0,:]=0
    # M_new[:,-1,:]=0
    # M_new[:,:,0]=0
    # M_new[:,:,-1]=0
    # N_new[:,0,:]=0
    # N_new[:,-1,:]=0
    # N_new[:,:,0]=0
    # N_new[:,:,-1]=0
    #N_new[0,:]=0
    #N_new[0,:]=0
    assert not torch.any(torch.isnan(M_new))
    assert not torch.any(torch.isnan(N_new))
    
    return M_new, N_new
    
def reconstruct_flow_depth_torch(H, Z, M, N, params):
    dry_limit = params.dry_limit
    # x方向通量
    z1_M = Z[:-1, :]
    z2_M = Z[1:, :]
    h1_M = H[1, :-1, :] #new H
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



def bcond_zeta(H, Z, params):
    """
    :ocean H:  domain elevation
    :ocean Z:  domain depth
    :param params: parameters structure containing:
        - obc_ele: boundary conditions
        - CC1: coefficient for x-direction
        - CC2: coefficient for y-direction
        - g: gravitational acceleration
    :return:
    """
    obc_ele = params.obc_ele
    CC1 = params.CC1
    CC2 = params.CC2
    g = params.g
    
    Nx, Ny = H.shape[1] , H.shape[2]
    # western side
    if obc_ele[0] == 'Cha_e': # explicit Chapman boundary condition
        for iy in range(Ny):
            Cx = CC1 * np.sqrt(g * ( Z[1, iy] + H[0, 1, iy]))
            H[1, 0, iy] = (1.0 - Cx) * H[0, 0, iy] + Cx * H[0, 1, iy]
    elif obc_ele[0] == 'Cha_i': # implicit Chapman boundary condition.
        for iy in range(Ny):
            Cx = CC1 * np.sqrt(g * ( Z[1, iy] + H[0, 1, iy]))
            cff2 = 1.0/(1.0+Cx)
            H[1, 0, iy] = cff2 * (H[0, 0, iy] + Cx * H[1, 1, iy])
    elif obc_ele[0] == 'Gra':
        H[1, 1, :] = H[1, 2, :]
    elif obc_ele[0] == 'Clo':
        H[1, 0, :] = 0.0
    elif obc_ele[0] == 'Rad':
        for iy in range(Ny):
            H[1, 0, iy] = H[0, 0, iy] - 2 * CC1 / np.sqrt(g * (Z[0, iy] + H[0, 0, iy])) * (H[0, 0, iy] - H[0, 1, iy])
    # eastern side
    if obc_ele[2] == 'Cha_e': # explicit Chapman boundary condition
        for iy in range(Ny):
            Cx = CC1 * np.sqrt(g * (Z[Nx - 2, iy] + H[0, Nx - 2, iy]))
            H[1, Nx - 1, iy] = (1. - Cx) * H[0, Nx - 1, iy] + Cx * H[0, Nx - 2, iy]
    elif obc_ele[2] == 'Cha_i':  # implicit Chapman boundary condition.
        for iy in range(Ny):
            Cx = CC1 * np.sqrt(g * (Z[Nx - 2, iy] + H[0, Nx - 2, iy]))
            cff2 = 1. / (1. + Cx)
            H[1, Nx - 1, iy] = cff2 * (H[0, Nx - 1, iy] + Cx * H[1, Nx - 2, iy])
    elif obc_ele[2] == 'Gra':
        H[1, Nx - 1, :] = H[1, Nx - 2, :]
    elif obc_ele[2] == 'Clo':
        H[1, Nx - 1, :] = 0.0
    elif obc_ele[2] == 'Rad':
        for iy in range(Ny):
            H[1, Nx - 1, iy] = H[0, Nx - 1, iy] - 2 * CC1 / np.sqrt(g * (Z[Nx - 1, iy] + H[0, Nx - 1, iy])) * (H[0, Nx - 1, iy] - H[0, Nx - 2, iy])

    # southern side
    if obc_ele[1] == 'Cha_e': # explicit Chapman boundary condition
        for ix in range(Nx):
            Ce = CC2 * np.sqrt(g * (Z[ix, 1] + H[0, ix, 1]))
            H[1, ix, 0] = (1. - Ce) * H[0, ix, 0] + Ce * H[0, ix, 1]
    elif obc_ele[1] == 'Cha_i':  # implicit Chapman boundary condition.
        for ix in range(Nx):
            Ce = CC2 * np.sqrt(g * (Z[ix, 1] + H[0, ix, 1]))
            cff2 = 1. / (1. + Ce)
            H[1, ix, 0] = cff2 * (H[0, ix, 0] + Ce * H[1, ix, 1])
    elif obc_ele[1] == 'Gra':
        H[1, :, 0] = H[1, :, 1]
    elif obc_ele[1] == 'Clo':
        H[1, :, 0] = 0.0
    elif obc_ele[1] == 'Rad':
        for ix in range(Nx):
            H[1, ix, 0] = H[0, ix, 0] - 2. * CC2 * np.sqrt(g * (Z[ix, 0] + H[0, ix, 0])) * (H[0, ix, 0] - H[0, ix, 1])
    # northern side
    if obc_ele[3] == 'Cha_e': # explicit Chapman boundary condition
        for ix in range(Nx):
            Ce = CC2 * np.sqrt(g * (Z[ix, Ny - 2] + H[0, ix, Ny - 2]))
            H[1, ix, Ny - 1] = (1. - Ce) * H[0, ix, Ny - 1] + Ce * H[0, ix, Ny - 2]
    elif obc_ele[3] == 'Cha_i':  # implicit Chapman boundary condition.
        for ix in range(Nx):
            Ce = CC2 * np.sqrt(g * (Z[ix, Ny - 2] + H[0, ix, Ny - 2]))
            cff2 = 1. / (1. + Ce)
            H[1, ix, Ny - 1] = cff2 * (H[0, ix, Ny - 1] + Ce * H[1, ix, Ny-2])
    elif obc_ele[3] == 'Gra':
        H[1, :, Ny - 1] = H[1, :, Ny - 2]
    elif obc_ele[3] == 'Clo':
        H[1, :, Ny - 1] = 0.0
    elif obc_ele[3] == 'Rad':
        for ix in range(Nx):
            H[1, ix, Ny - 1] = H[0, ix, Ny - 1] - 2. * CC2 * np.sqrt(g * (Z[ix, Ny - 1] + H[0, ix, Ny - 1])) * (H[0, ix, Ny - 1] - H[0, ix, Ny - 2])
    return H

def bcond_u2D(H, Z, M, D_M, z_w, z_e, params):
    """
    :param H:  domain elevation
    :param Z:  depth
    :param z_e:  elevation in east bound
    :param z_w:  elevation in west bound
    :param params: parameters structure containing:
        - dt: time step
        - dy: y grid spacing
        - dx: x grid spacing
        - g: gravitational acceleration
        - CC1: coefficient for western boundary
        - CC2: coefficient for southern/northern boundary
                    # W S E N
        - obc_u2d = ['Fla', 'Clo', 'Gra', 'Clo']
    """
    obc_u2d = params.obc_u2d
    Nx, Ny = H.shape[1], H.shape[2]  # 此处： L = Nx, M = Ny, Lm = Nx - 1, Mm = Ny - 1
    # Southern side
    if obc_u2d[1] == 'Fla':
        ubar_s = np.zeros((Nx - 1, 1), dtype=float)
        for ix in range(Nx-1):
            cff = params.dt * 0.5 / params.dy
            cff1 = np.sqrt(params.g * 0.5 * (Z[ix, 1] + H[0, ix, 1] + Z[ix + 1, 1] + H[0, ix + 1, 1]))
            Ce = cff * cff1
            cff2 = 1.0 / (1.0 + Ce)
            ubar_s[ix] = cff2 * (M[0, ix, 0] / D_M[ix, 0] + Ce * M[1, ix, 1] / D_M[ix, 1])
            M[1, ix, 0] = ubar_s[ix] * D_M[ix, 0]
    elif obc_u2d[1] == 'Gra':
        M[1, :, 0] = M[1, :, 1]
    elif obc_u2d[1] == 'Clo':
        M[1, :, 0] = 0.0
    if obc_u2d[1] == 'Rad':
        ubar_s = np.zeros((Nx - 1, 1), dtype=float)
        for ix in range(Nx-1):
            cff = Z[ix, 0] + H[0, ix, 0] + Z[ix + 1, 0] + H[0, ix + 1, 0]
            cff1 = M[0, ix, 0] / D_M[ix, 0] - M[0, ix, 1] / D_M[ix, 1]
            cff2 = Z[ix, 0] + H[1, ix, 0] + Z[ix + 1, 0] + H[1, ix + 1, 0]
            ubar_s[ix] = M[0, ix, 0] / D_M[ix, 0] * cff - 2 * params.CC2 * np.sqrt(params.g * cff * 0.5) * cff1
            ubar_s[ix] = ubar_s[ix] / cff2
            M[1, ix, 0] = ubar_s[ix] * D_M[ix, 0]
    # northern side
    if obc_u2d[3] == 'Fla':
        ubar_n = np.zeros((Nx - 1, 1), dtype=float)
        for ix in range(Nx-1):
            cff = params.dt * 0.5 / params.dy
            cff1 = np.sqrt(params.g * 0.5 * (Z[ix, Ny - 2] + H[0, ix, Ny - 2] + Z[ix + 1, Ny - 2] + H[0, ix + 1, Ny - 2]))
            Ce = cff * cff1
            cff2 = 1.0 / (1.0 + Ce)
            ubar_n[ix] = cff2 * (M[0, ix, Ny - 1] / D_M[ix, Ny - 1] + Ce * M[1, ix, Ny - 2] / D_M[ix, Ny - 2])
            M[1, ix, Ny - 1] = ubar_n[ix] * D_M[ix, Ny - 1]
    elif obc_u2d[3] == 'Gra':
        M[1, :, Ny-1] = M[1, :, Ny-2]
    elif obc_u2d[3] == 'Clo':
        M[1, :, Ny-1] = 0.0
    if obc_u2d[3] == 'Rad':
        ubar_n = np.zeros((Nx - 1, 1), dtype=float)
        for ix in range(Nx-1):
            cff = Z[ix, Ny - 1] + H[0, ix, Ny - 1] + Z[ix + 1, Ny - 1] + H[0, ix + 1, Ny - 1]
            cff1 = M[0, ix, Ny - 1] / D_M[ix, Ny - 1] - M[0, ix, Ny - 2] / D_M[ix, Ny - 2]
            cff2 = Z[ix, Ny - 1] + H[1, ix, Ny - 1] + Z[ix + 1, Ny - 1] + H[1, ix + 1, Ny - 1]
            ubar_s[ix] = M[0, ix, Ny - 1] / D_M[ix, Ny - 1] * cff - 2 * params.CC2 * np.sqrt(params.g * cff * 0.5) * cff1
            ubar_s[ix] = ubar_s[ix] / cff2
            M[1, ix, Ny - 1] = ubar_n[ix] * D_M[ix, Ny - 1]
    # western side
    if obc_u2d[0] == 'Fla':
        ubar_w = np.zeros((Ny, 1), dtype=float)
        for iy in range(Ny):
            bry_pgr = -params.g * (H[0, 1, iy] - z_w[iy]) * 0.5 * params.CC1
            bry_cor = 0.0 # should add
            cff1 = 1.0 / (0.5 * (Z[0, iy] + H[0, 0, iy] + Z[1, iy] + H[0, 1, iy]))
            bry_str = 0.0 # should add surface forcing
            Cx = 1.0 / np.sqrt(params.g * 0.5 * (Z[0, iy] + H[0, 0, iy] + Z[1, iy] + H[0, 1, iy]))
            cff2 = Cx / params.dx
            bry_val = M[0, 1, iy] / D_M[1, iy] + cff2 * (bry_pgr + bry_cor + bry_str)
            Cx = np.sqrt( params.g * cff1 )
            ubar_w[iy] = bry_val - Cx * ( 0.5 * (H[0, 0, iy] + H[0, 1, iy]) - z_w[iy])
            M[1, 0, iy] = np.float64(ubar_w[iy]) * D_M[0, iy]
    elif obc_u2d[0] == 'Gra':
        M[1, 0, :] = M[1, 1, :]
    elif obc_u2d[0] == 'Clo':
        M[1, 0, :] = 0.0
    elif obc_u2d[0] == 'Rad':
        ubar_w = np.zeros((Ny, 1), dtype=float)
        for iy in range(Ny):
            ubar_w[iy] = -1.0 * np.sqrt(params.g / (Z[0, iy] + H[1, 0, iy])) * H[1, 0, iy]
            M[1, 0, iy] = np.float64(ubar_w[iy]) * D_M[0, iy]
    # Eastern side
    if obc_u2d[2] == 'Fla':
        ubar_e = np.zeros((Ny, 1), dtype=float)
        for iy in range(Ny):
            bry_pgr = -g * (z_e[iy] - H[0, Nx - 3, iy]) * 0.5 * CC1
            bry_cor = 0.0  # should add
            cff1 = 1.0 / (0.5 * (Z[Nx - 3, iy] + H[0, Nx - 3, iy] + Z[Nx - 2, iy] + H[0, Nx - 2, iy]))
            bry_str = 0.0  # should add surface forcing
            Cx = 1.0 / np.sqrt(g * 0.5 * (Z[Nx - 2, iy] + H[0, Nx - 2, iy] + Z[Nx - 3, iy] + H[0, Nx - 3, iy]))
            cff2 = Cx / dx
            bry_val = M[0, Nx - 3, iy] / D_M[Nx - 3, iy] + cff2 * (bry_pgr + bry_cor + bry_str)
            Cx = np.sqrt(g * cff1)
            ubar_e[iy] = bry_val + Cx * (0.5 * (H[0, Nx - 3, iy] + H[0, Nx - 2, iy]) - z_e[iy])
            M[1, Nx - 2, iy] = ubar_e[iy] * D_M[Nx - 2, iy]
    elif obc_u2d[2] == 'Gra':
        M[1, Nx - 2, :] = M[1, Nx - 3, :]
    elif obc_u2d[2] == 'Clo':
        M[1, Nx - 2, :] = 0.0
    elif obc_u2d[2] == 'Rad':
        ubar_e = np.zeros((Ny, 1), dtype=float)
        for iy in range(Ny):
            ubar_e[iy] = 1.0 * np.sqrt(g / (Z[Nx - 1, iy] + H[1, Nx - 1, iy])) * H[1, Nx - 1, iy]
            M[1, Nx - 2, iy] = ubar_e[iy] * D_M[Nx - 2, iy]

    return M

def bcond_v2D(H, Z, N, D_N, z_s, z_n, Params):
    """
    :param H:  domain elevation
    :param Z:  depth
    :param z_s:  elevation in south bound
    :param z_n:  elevation in north bound
    :param params: parameters structure containing:
        - dt: time step
        - dy: y grid spacing
        - dx: x grid spacing
        - g: gravitational acceleration
        - CC1: coefficient for western boundary
        - CC2: coefficient for southern/northern boundary
                    # W S E N
        - obc_v2d = ['Fla', 'Clo', 'Gra', 'Clo']
    """
    obc_v2d = params.obc_v2d
    Nx, Ny = H.shape[1], H.shape[2]
    
    # western side
    if obc_v2d[0] == 'Fla':
        vbar_w = np.zeros((Ny - 1, 1), dtype=float)
        for iy in range(Ny-1):
            cff = params.dt * 0.5 / params.dx
            cff1 = np.sqrt(params.g * 0.5 * (Z[1, iy] + H[0, 1, iy] + Z[1, iy + 1] + H[0, 1, iy + 1]))
            Cx = cff * cff1
            cff2 = 1.0 / (1.0 + Cx)
            vbar_w[iy] = cff2 * (N[0, 0, iy] / D_N[0, iy] + Cx * N[1, 1, iy] / D_N[1, iy])
            N[1, 0, iy] = vbar_w[iy] * D_N[0, iy]
    elif obc_v2d[0] == 'Gra':
        N[1, 0, :] = N[1, 1, :]
    elif obc_v2d[0] == 'Clo':
        N[1, 0, :] = 0.0
    elif obc_v2d[0] == 'Rad':
        vbar_w = np.zeros((Ny - 1, 1), dtype=float)
        for iy in range(Ny - 1):
            cff = Z[0, iy] + H[0, 0, iy] + Z[0, iy + 1] + H[0, 0, iy + 1]
            cff1 = N[0, 0, iy] / D_N[0, iy] - N[0, 1, iy] / D_N[1, iy]
            cff2 = Z[0, iy] + H[1, 0, iy] + Z[0, iy + 1] + H[1, 0, iy + 1]
            vbar_w[iy] = N[0, 0, iy] / D_N[0, iy] * cff - 2 * params.CC1 * np.sqrt(params.g * cff * 0.5) * cff1
            vbar_w[iy] = vbar_w[iy] / cff2
            N[1, 0, iy] = vbar_w[iy] * D_N[0, iy]
    
    # Eastern side
    if obc_v2d[2] == 'Fla':
        vbar_e = np.zeros((Ny - 1, 1), dtype=float)
        for iy in range(Ny - 1):
            cff = params.dt * 0.5 / params.dx
            cff1 = np.sqrt(params.g * 0.5 * (Z[Nx - 2, iy] + H[0, Nx - 2, iy] + Z[Nx - 2, iy + 1] + H[0, Nx - 2, iy + 1]))
            Cx = cff * cff1
            cff2 = 1.0 / (1.0 + Cx)
            vbar_e[iy] = cff2 * (N[0, Nx - 1, iy] / D_N[Nx - 1, iy] + Cx * N[1, Nx - 2, iy] / D_N[Nx - 2, iy])
            N[1, Nx - 1, iy] = vbar_e[iy] * D_N[Nx - 1, iy]
    elif obc_v2d[2] == 'Gra':
        N[1, Nx - 1, :] = N[1, Nx - 2, :]
    elif obc_v2d[2] == 'Clo':
        N[1, Nx - 1, :] = 0.0
    elif obc_v2d[2] == 'Rad':
        vbar_e = np.zeros((Ny - 1, 1), dtype=float)
        for iy in range(Ny - 1):
            cff = Z[Nx - 1, iy] + H[0, Nx - 1, iy] + Z[Nx - 1, iy + 1] + H[0, Nx - 1, iy + 1]
            cff1 = N[0, Nx - 1, iy] / D_N[Nx - 1, iy] - N[0, Nx - 2, iy] / D_N[Nx - 2, iy]
            cff2 = Z[Nx - 1, iy] + H[1, Nx - 1, iy] + Z[Nx - 1, iy + 1] + H[1, Nx - 1, iy + 1]
            vbar_e[iy] = N[0, Nx - 1, iy] / D_N[Nx - 1, iy] * cff - 2 * params.CC1 * np.sqrt(params.g * cff * 0.5) * cff1
            vbar_e[iy] = vbar_e[iy] / cff2
            N[1, Nx - 1, iy] = vbar_e[iy] * D_N[Nx - 1, iy]
    
    # Southern side
    if obc_v2d[1] == 'Fla':
        vbar_s = np.zeros((Nx, 1), dtype=float)
        for ix in range(Nx):
            bry_pgr = -params.g * (H[0, ix, 0] - z_s[ix]) * 0.5 * params.CC2
            bry_cor = 0.0
            cff1 = 1.0 / (0.5 * (Z[ix, 0] + H[0, ix, 0] + Z[ix, 1] + H[0, ix, 1]))
            bry_str = 0.0
            Ce = 1.0 / np.sqrt(params.g * 0.5 * (Z[ix, 0] + H[0, ix, 0] + Z[ix, 1] + H[0, ix, 1]))
            cff2 = Ce / params.dy
            bry_val = N[0, ix, 1] / D_N[ix, 1] + cff2 * (bry_pgr + bry_cor + bry_str)
            Ce = np.sqrt(params.g * cff1)
            vbar_s[ix] = bry_val - Ce * (0.5 * (H[0, ix, 0] + H[0, ix, 1]) - z_s[ix])
            N[1, ix, 0] = vbar_s[ix] * D_N[ix, 0]
    elif obc_v2d[1] == 'Gra':
        N[1, :, 0] = N[1, :, 1]
    elif obc_v2d[1] == 'Clo':
        N[1, :, 0] = 0.0
    elif obc_v2d[1] == 'Rad':
        vbar_s = np.zeros((Nx, 1), dtype=float)
        for ix in range(Nx):
            vbar_s[ix] = -1.0 * np.sqrt(params.g / (Z[ix, 0] + H[1, ix, 0])) * H[1, ix, 0]
            N[1, ix, 0] = vbar_s[ix] * D_N[ix, 0]
    
    # northern side
    if obc_v2d[3] == 'Fla':
        vbar_n = np.zeros((Nx, 1), dtype=float)
        for ix in range(Nx):
            bry_pgr = -params.g * (z_n[ix] - H[0, ix, Ny -3]) * 0.5 * params.CC2
            bry_cor = 0.0
            bry_pgr = -g * (z_n[ix] - H[0, ix, Ny -3]) * 0.5 * CC2
            bry_cor = 0.0  # should add
            cff1 = 1.0 / (0.5 * (Z[ix, Ny - 3] + H[0, ix, Ny - 3] + Z[ix, Ny - 2] + H[0, ix, Ny - 2]))
            bry_str = 0.0  # should add surface forcing
            Ce = 1.0 / np.sqrt(g * 0.5 * (Z[ix, Ny - 2] + H[0, ix, Ny - 2] + Z[ix, Ny - 3] + H[0, ix, Ny - 3]))
            cff2 = Ce / dy
            bry_val = N[0, ix, Ny - 3] / D_N[ix, Ny - 3] + cff2 * (bry_pgr + bry_cor + bry_str)
            Ce = np.sqrt(g * cff1)
            vbar_n[ix] = bry_val + Ce * (0.5 * (H[0, ix, Ny - 3] + H[0, ix, Ny - 2]) - z_n[ix])
            N[1, ix, Ny - 2] = vbar_n[ix] * D_N[ix, Ny-2]
    elif obc_v2d[3] == 'Gra':
        N[1, :, Ny - 2] = N[1, :, Ny - 3]
    elif obc_v2d[3] == 'Clo':
        N[1, :, Ny - 2] = 0.0
    elif obc_v2d[3] == 'Rad':
        vbar_n = np.zeros((Nx, 1), dtype=float)
        for ix in range(Nx):
            vbar_n[ix] = 1.0 * np.sqrt(g / (Z[ix, Ny - 1] + H[1, ix, Ny - 1])) * H[1, ix, Ny - 1]
            N[1, ix, Ny - 2] = vbar_n[ix] * D_N[ix, Ny - 2]

    return N

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


if __name__ == '__main__':
    params = Params()
    device = torch.device('cpu')
    
    dt = params.dt
    NT = params.NT
    
    x = torch.linspace(0, params.Lx, params.Nx + 1)
    y = torch.linspace(0, params.Ly, params.Ny + 1)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    M = torch.zeros((2, params.Nx, params.Ny+1))
    N = torch.zeros((2, params.Nx+1, params.Ny))
    H = torch.zeros((2, params.Nx+1, params.Ny+1))
    Z = torch.ones((params.Nx+1, params.Ny+1)) * params.depth
    
    H[0] = 1 * torch.exp(-((X-400)**2 + (Y-400)**2) / (2 * 5.0**2))
    #
    #Wx,Wy,Pa = generate_wind(X,Y,params)
    #Ws = torch.sqrt(Wx**2 + Wy**2)
    #
    Wx = np.ones((NT,X.shape[0],X.shape[1]))*0 + 10
    Wy = np.ones((NT,X.shape[0],X.shape[1]))*0
    
    Wx = torch.from_numpy(Wx)
    Wy = torch.from_numpy(Wy)
    Pa = Wx * 0 + 1000
    
    eta_list = list()
    u_list = list()
    v_list = list()
    anim_interval = 1
    
    for itime in range(params.NT):
        print(f"nt = {itime} / {params.NT}")
        params.itime=itime
        H_update = mass_cartesian_torch(H, Z, M, N, params)
        
        M_update, N_update = momentum_nonlinear_cartesian_torch(H_update, Z, M, N, Wx[itime], Wy[itime], Pa[itime], params)

        H[0] = H_update[1]
        M[0] = M_update[1]
        N[0] = N_update[1]
        eta_list.append(dd(H_update[1]))
        u_list.append(dd(M_update[1]))
        v_list.append(dd(N_update[1]))
        
        pcolor(X, Y, eta_list[-1], vmin=-.2,vmax=.2, cmap=plt.cm.RdBu_r)
        colorbar()
        #quiver(X[:-1,:-1],Y[:-1,:-1],rho2v(M_update[1]),rho2u(N_update[1]), scale=10)
        xlabel("x [m]", fontname="serif", fontsize=12)
        ylabel("y [m]", fontname="serif", fontsize=12)
        title("Stage at an instant in time: " + f"{itime*params.dt}" + " second")
        axis('equal')
        #fig.savefig("ele_Rad" + f"{itime}" + "s.jpg", format='jpg', dpi=300, bbox_inches='tight')
        show()
        
    
    
    ds_out = xr.Dataset(
            {'eta': (['time','lat', 'lon'], np.array(eta_list))},
            coords={
                'lat': x,
                'lon': y,
                'time': np.arange(len(eta_list))}  )
    ds_out.to_netcdf('out.circulation.nc')
    
    eta = np.array(eta_list)
    ds = xr.open_dataset('rst.channel.Move3050.nc')
    eta_true = ds['zeta'][:]
    eta_true = np.swapaxes(eta_true,1,2)
    time2plot = [1, itime//10, itime//5, itime//3, itime//2, itime-2]
    for itime in time2plot:
        ind2plot = itime
        fig, ax = plt.subplots(1, 1)
        # plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 17)
        xlabel("x [m]", fontname="serif", fontsize=12)
        ylabel("y [m]", fontname="serif", fontsize=12)
        title("Stage at an instant in time: " + f"{itime*dt}" + " second")
    # pmesh = plt.pcolormesh(X, Y, eta_list[500], vmin=-0.7 * np.abs(eta_list[int(len(eta_list) / 2)]).max(),
    #                        vmax=np.abs(eta_list[int(len(eta_list) / 2)]).max(), cmap=plt.cm.RdBu_r)
        ubar = u_list[ind2plot]
        vbar = v_list[ind2plot]
        pcolor(X, Y, eta_list[ind2plot], vmin=-.2,vmax=.2, cmap=plt.cm.RdBu_r)
        colorbar()
        quiver(X[::3,::3],Y[::3,::3],ubar[::3,::3],vbar[::3,::3], scale=70)
        #quiver(X[::3,::3],Y[::3,::3],(Wx/Ws)[ind2plot,::3,::3],(Wy/Ws)[ind2plot,::3,::3], scale=70)
        axis('equal')
        savefig("ele_Rad" + f"{itime*dt}" + "s.jpg", format='jpg', dpi=300, bbox_inches='tight')
        #show()
        figure()
        plot(eta[ind2plot,:,25])
        plot(eta_true[ind2plot,:,25],'r--')
        legend(('Model','ROMS'))
        title("Profile at Y=200km at time: " + f"{itime*dt}" + " second")
        savefig("ele_profile " + f"{itime*dt}" + "s.jpg", format='jpg', dpi=300, bbox_inches='tight')
        
        
        
    # x2plot = [2, 10, 30, 50, 70, 90, 95, 120, 150, 180, 220]
    x2plot = [20, 30]
    for ix in x2plot:
        plot_stage_Point(eta_list, ix)
    # fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    # plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname="serif", fontsize=19)
    # plt.xlabel("x [km]", fontname="serif", fontsize=16)
    # plt.ylabel("y [km]", fontname="serif", fontsize=16)
    # x_int = 10
    # y_int = 5
    # xx = X[:-1:x_int, ::y_int]
    # yy = Y[:-1:x_int, ::y_int]
    # uu = u_list[-1][::x_int, ::y_int]
    # vv = v_list[-1][:-1:x_int, ::y_int]
    # Q = ax.quiver(xx , yy, uu,vv, scale=10, scale_units='inches')
    # fig.savefig("vec.jpg", format='jpg', dpi=300, bbox_inches='tight')
    # plt.close()
    #eta_anim = vis_tools.eta_animation(X, Y, eta_list, anim_interval*dt, "eta")