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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mass_cartesian_torch(H0, Z, M0, N0, params, manning):
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
        
    Nx, Ny = H0.shape
      
    dMdx = ddx(M0)
    dNdy = ddy(N0)
    
    H1 = H0.clone()
    H1[1:-1,1:-1] = H0[1:-1,1:-1].detach() - CC1 * dMdx[1:-1,1:-1] - CC2 * dNdy[1:-1,1:-1]
        
    #TODO    
    # 干湿修正（使用 torch.where 保证全为新张量）
    mask_deep = (Z <= -dry_limit).to(H1.device)
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
    #TODO boundary tide
    #H_new[1,1,:] = 1.0 * np.sin(2 * np.pi / 200 * params.itime * params.dt)
    
    H1 = bcond_zeta_torch(H1, Z, params,H0=None) #if using Chapman, H0 need to be provided
    
    assert not torch.any(torch.isnan(H1))
    
    return H1


def momentum_nonlinear_cartesian_simple(H, Z, M, N, Wx, Wy, Pa, params, manning):
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
    Cf = manning
    dt = params.dt
    phi = params.centerWeighting0
    Nx = params.Nx
    Ny = params.Ny
        
    # 重构流深, TODO(check Lechi的修改)
    D_M, D_N = reconstruct_flow_depth_torch(H, Z, M, N, params)

    # get forcing
    windSpeed = torch.sqrt(Wx * Wx + Wy * Wy)
    sustr = rho2u( rho_air / rho_water * Cd * windSpeed * Wx)
    svstr = rho2v( rho_air / rho_water * Cd * windSpeed * Wy)
    
    # 以下是何乐驰版本
    # ========== M 分量 (Nx-1, Ny) ==========
    m0 = M[0].clone().detach()
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

    D0 = D0.to(device)
    dPdx = dPdx.to(device)

    
    Pre_grad_x = CC1 * D0 * dPdx / rho_water
    
    ududx = F.pad( ududx_up(M[0],N[0],Z+H[1]), (0,0,1,1)) #pad for up and down
    vdudy = F.pad( vdudy_up(M[0],N[0],Z+H[1]), (1,1,0,0)) #pad for left and right
    
    # Nu is applied here and in the friction below
    N_exp = torch.cat( (N[0,:,0:1], N[0], N[0,:,-1:]), dim = 1)
    Nu = rho2u(v2rho(N_exp))   #at u-point
    
    # 底摩擦
    mask_fs999_M = (flux_sign_M == 999)
    #M_val = torch.where(mask_fs999_M, torch.zeros_like(M_val), M_val)
    #by LWF
    friction_mask = (D0 > FrictionDepthLimit)
    epsilon = 1e-9
    Cf_u = rho2u(manning)

    #Nu was generated before
    Fx = g * Cf_u**2 / (D0**2.33 + 1e-9) * torch.sqrt(m0**2 + Nu**2) * m0
    #by LWF, when optimize, do not use clamping, replacing, or non-torch operations
    # Fx = g * Cf_u**2 / (D0**2.33 + 1e-9) * torch.sqrt(torch.clamp(m0**2 + Nu**2, min=epsilon)) * m0
    # Fx = torch.where(torch.abs(dt*Fx)>torch.abs(m0), m0/dt, Fx)
    # Fx = torch.where(friction_mask, Fx, torch.zeros_like(Fx))
    sustr = torch.tensor(sustr, dtype=torch.float64).to(device)
    f_cor = torch.tensor(f_cor, dtype=torch.float64).to(device)
    Nu = torch.tensor(Nu, dtype=torch.float64).to(device)    
    
    phi_M = 1.0 - CC1 * torch.min(torch.abs(m0 / torch.clamp(D0, min=MinWaterDepth)), torch.sqrt(g*D0))
    M_new = M.clone().detach() #by LWF    
    # M_new[1] = (phi_M*m0 + 0.5*(1.0 - phi_M)*(m1 + m2)        #implicit friction
    #         + dt  * ( sustr + f_cor * Nu)                  #wind stress and coriolis
    #         - CC1 * ududx                                  #u advection
    #         - CC2 * vdudy                                  #v advection
    #         - CC3 * D0*(h2u_M - h1u_M) + Pre_grad_x        #pgf
    #         - dt  * Fx)                                    #friction
    
    M_new[1] = dt  * Fx
    #pcolor(rho2u(X),rho2u(Y),ududx);colorbar();show()
    #pcolor(rho2u(X),rho2u(Y),vdudy);colorbar();show()

    # ========== N 分量 (Nx, Ny-1) ==========
    n0 = N[0].clone().detach()
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
    
    D0N = D_N.to(device)
    dPdy = dPdy.to(device)


    Pre_grad_y = CC2 * D0N * dPdy / rho_water
    
    udvdx = F.pad( udvdx_up(M[0],N[0],Z+H[1]), (0,0,1,1)) #pad for up and down
    vdvdy = F.pad( vdvdy_up(M[0],N[0],Z+H[1]), (1,1,0,0)) #pad for left and right
    
    # Mv is applied here and in the friction below
    M_exp = torch.cat( (M[0,0:1],M[0], M[0,-1:]), dim =0)
    Mv = rho2u(v2rho(M_exp)) #at v-point, [ 1 ~ Nx-1, 1 ~ Ny-1]
    #Mv = F.pad( rho2u(rho2v( M[0])) , (0,0,1,1)) #pad for up and down
    #底摩擦
    mask_fs999_N = (flux_sign_N == 999)
    #N_val = torch.where(mask_fs999_N, torch.zeros_like(N_val), N_val)
    friction_maskN = (D0N > FrictionDepthLimit)
    epsilon = 1e-9
    Cf_v = rho2v(Cf)

    
    Fy = g * Cf_v**2 / (D0N**2.33 + 1e-9) * torch.sqrt(Mv**2 + n0**2) * n0
    #by LWF, when optimize, do not use clamping, replacing, or non-torch operations
    #Fy = g * Cf_v**2 / (D0N**2.33 + 1e-9) * torch.sqrt(torch.clamp(Mv**2 + n0**2, min=epsilon)) * n0
    #Fy = torch.where(torch.abs(dt*Fy)>torch.abs(n0), n0/dt, Fy)
    #Fy = torch.where(friction_maskN, Fy, torch.zeros_like(Fy))
    # 
    svstr = torch.tensor(svstr, dtype=torch.float64).to(device)

    Mv = torch.tensor(Mv, dtype=torch.float64).to(device) 
    
    N_new = N.detach() #by LWF
    phi_N = 1.0 - CC2 * torch.min(torch.abs(n0 / torch.clamp(D0N, min=MinWaterDepth)), torch.sqrt(g*D0N))
    N_new[1] = (phi_N*n0 + 0.5*(1.0 - phi_N)*(n1 + n2) 
            + dt * (svstr - f_cor * Mv )
            - CC1 * udvdx
            - CC2 * vdvdy
            - CC4 * D0N * (h2u_N - h1u_N) + Pre_grad_y
            - dt * Fy )
    #pcolor(rho2v(X)[1:-1,1:-1],rho2v(Y)[1:-1,1:-1],N_val[1:-1,1:-1]/D_N[1:-1,1:-1]);colorbar();show()
    
    # apply boundary condition
    z_w = 0 #torch.from_numpy(0.5 * np.sin(2 * np.pi / 5 * itime * dt) * (np.zeros((Ny + 1, 1)) + 1))
    z_e = 0
    z_n = 0
    z_s = 0
    #TODO check       
    M_new = bcond_u2D_torch(H, Z, M_new, D_M, z_w, z_e, params)
    N_new = bcond_v2D_torch(H, Z, N_new, D_N, z_s, z_n, params)
    assert not torch.any(torch.isnan(M_new))
    assert not torch.any(torch.isnan(N_new))
    
    return M_new, N_new, Fx, Fy
    
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
    flux_sign = torch.zeros_like(z1, dtype=torch.int32, device=device)

    cond1 = (z1 <= -dry_limit) | (z2 <= -dry_limit)
    flux_sign = torch.where(cond1, torch.tensor(999, dtype=torch.int32, device=device), flux_sign)

    cond2 = ((z1 + h1_out) <= 0) | ((z2 + h2_out) <= 0)
    cond2a = ((z1 + h1_out) <= 0) & ((z2 + h2_out) <= 0)
    cond2b = (z1 > z2) & (z1 + h1_out > 0) & ((z2 + h1_out) <= 0)
    cond2c = (z2 > z1) & (z2 + h2_out > 0) & ((z1 + h2_out) <= 0)
    cond2_any = cond2 & (cond2a | cond2b | cond2c)
    flux_sign = torch.where(cond2_any, torch.tensor(999, dtype=torch.int32, device=device), flux_sign)

    cond3 = (z1 <= z2) & (z1 + h1_out > 0) & ((z1 + h2_out) <= 0)
    flux_sign = torch.where(cond3, torch.tensor(1, dtype=torch.int32, device=device), flux_sign)
    h2_out = torch.where(cond3, -z1, h2_out)

    cond4 = (z1 >= z2) & ((z2 + h2_out) <= 0) & (z2 + h1_out > 0)
    flux_sign = torch.where(cond4, torch.tensor(2, dtype=torch.int32, device=device), flux_sign)
    h2_out = torch.where(cond4, -z2, h2_out)

    cond5 = (z2 <= z1) & (z2 + h2_out > 0) & ((z2 + h1_out) <= 0)
    flux_sign = torch.where(cond5, torch.tensor(-1, dtype=torch.int32, device=device), flux_sign)
    h1_out = torch.where(cond5, -z2, h1_out)

    cond6 = (z2 >= z1) & ((z1 + h1_out) <= 0) & (z1 + h2_out > 0)
    flux_sign = torch.where(cond6, torch.tensor(-2, dtype=torch.int32, device=device), flux_sign)
    h1_out = torch.where(cond6, -z1, h1_out)

    return flux_sign, h1_out, h2_out



def bcond_zeta_vec(H, Z, params, tide_z):
    """
    :ocean H1: domain elevation (PyTorch tensor)
    :ocean H0: optional, elevation of last step
    :ocean Z:  domain depth (PyTorch tensor)
    :param params: parameters structure containing:
        - obc_ele: boundary conditions
        - CC1: coefficient for x-direction
        - CC2: coefficient for y-direction
        - g: gravitational acceleration
    :return:
    """
    H1=H[1]
    H0=H[0]
    obc_ele = params.obc_ele
    CC1 = params.CC1
    CC2 = params.CC2
    g = params.g
    Nx, Ny = H1.shape[0], H1.shape[1]
    # western side
    if obc_ele[0] == 'Cha_e':  # explicit Chapman boundary condition
        
        if CC1.dim()>1: # CC1 is a matrix, ie, grid length is variable
            Cx = CC1[1] * torch.sqrt(g * (Z[1] + H0[1]))
        else: # CC1 is a number, i.e., grid length is a constant
            Cx = CC1    * torch.sqrt(g * (Z[1] + H0[1]))
            
        H1[0] = (1.0 - Cx) * H0[0] + Cx * H0[1]
    elif obc_ele[0] == 'Cha_i':  # implicit Chapman boundary condition.
        #for iy in range(Ny):
        if CC1.dim()>1: # CC1 is a matrix, ie, grid length is variable
            Cx = CC1[1] * torch.sqrt(g * (Z[1] + H0[1]))
        else: # CC1 is a number, i.e., grid length is a constant
            Cx = CC1    * torch.sqrt(g * (Z[1] + H0[1]))
        
        cff2 = 1.0 / (1.0 + Cx)
        H1[0] = cff2 * (H0[0] + Cx * H1[1])
    elif obc_ele[0] == 'Gra':
        H1[0] = H1[1] # by LWF
    elif obc_ele[0] == 'Clo':
        H1[0] = 0.0
    elif obc_ele[0] == 'Rad':
        #for iy in range(Ny): removing the Ny loop
        H1[0] = H0[0] - 2 * CC1 / torch.sqrt(g * (Z[0] + H0[0])) * (H0[0] - H0[1])
    
    # --- Eastern Boundary (iy loop -> vectorized) ---
    if obc_ele[2] == 'Cha_e':
        if CC1.dim()>1: # CC1 is a matrix, ie, grid length is variable
            Cx = CC1[Nx-2] * torch.sqrt(g * (Z[Nx-2] + H0[Nx-2]))
        else: # CC1 is a number, i.e., grid length is a constant
            Cx = CC1       * torch.sqrt(g * (Z[Nx-2] + H0[Nx-2]))
            
        H1[Nx-1, :] = (1.0 - Cx) * H0[Nx-1, :] + Cx * H0[Nx-2, :]
    elif obc_ele[2] == 'Cha_i':
        # Same with Cha_e
        if CC1.dim()>1: # CC1 is a matrix, ie, grid length is variable
            Cx = CC1[Nx-2] * torch.sqrt(g * (Z[Nx-2] + H0[Nx-2]))
        else: # CC1 is a number, i.e., grid length is a constant
            Cx = CC1       * torch.sqrt(g * (Z[Nx-2] + H0[Nx-2]))
        cff2 = 1.0 / (1.0 + Cx)
        H1[Nx-1, :] = cff2 * (H0[Nx-1, :] + Cx * H1[Nx-2, :])
    elif obc_ele[2] == 'Gra':
        H1[Nx-1, :] = H1[Nx-2, :]
    elif obc_ele[2] == 'Clo':
        H1[Nx-1, :] = 0.0
    elif obc_ele[2] == 'Rad':
        if CC1.dim()>1: # CC1 is a matrix, ie, grid length is variable
            Cx = 2 * CC1[Nx-1] / torch.sqrt(g * (Z[Nx-1, :] + H0[Nx-1, :]))
        else: # CC1 is a number, i.e., grid length is a constant
            Cx = 2 * CC1      / torch.sqrt(g * (Z[Nx-1, :] + H0[Nx-1, :]))
            
        H1[Nx-1, :] = H0[Nx-1, :] - Cx * (H0[Nx-1, :] - H0[Nx-2, :])
        
    # --- Southern Boundary (ix loop -> vectorized) ---
    if obc_ele[1] == 'Cha_e':
        #
        if CC2.dim()>1: # CC1 is a matrix, ie, grid length is variable
            Ce = CC2[:,1] * torch.sqrt(g * (Z[:, 1] + H0[:, 1]))
        else: # CC1 is a number, i.e., grid length is a constant
            Ce = CC2 * torch.sqrt(g * (Z[:, 1] + H0[:, 1]))
        
        H1[:, 0] = (1.0 - Ce) * H0[:, 0] + Ce * H0[:, 1]
    elif obc_ele[1] == 'Cha_i':
        #
        if CC2.dim()>1: # CC2 is a matrix, ie, grid length is variable
            Ce = CC2[:,1] * torch.sqrt(g * (Z[:, 1] + H0[:, 1]))
        else: # CC2 is a number, i.e., grid length is a constant
            Ce = CC2 * torch.sqrt(g * (Z[:, 1] + H0[:, 1]))
        
        cff2 = 1.0 / (1.0 + Ce)
        H1[:, 0] = cff2 * (H0[:, 0] + Ce * H1[:, 1])
    elif obc_ele[1] == 'Gra':
        H1[:, 0] = H1[:, 1]
    elif obc_ele[1] == 'Clo':
        H1[:, 0] = 0.0
    elif obc_ele[1] == 'Rad':
        #
        if CC2.dim()>1: # CC1 is a matrix, ie, grid length is variable
            Ce = 2.0 * CC2[:,0] * torch.sqrt(g * (Z[:, 0] + H0[:, 0]))
        else: # CC1 is a number, i.e., grid length is a constant
            Ce = 2.0 * CC2      * torch.sqrt(g * (Z[:, 0] + H0[:, 0]))
        #    
        H1[:, 0] = H0[:, 0] - Ce * (H0[:, 0] - H0[:, 1])
    
    # --- Northern Boundary (ix loop -> vectorized) ---
    if obc_ele[3] == 'Cha_e':
        #
        if CC2.dim()>1: # CC2 is a matrix, ie, grid length is variable
            Ce = CC2         * torch.sqrt(g * (Z[:, Ny-2] + H0[:, Ny-2]))
        else: # CC2 is a number, i.e., grid length is a constant
            Ce = CC2[:,Ny-2] * torch.sqrt(g * (Z[:, Ny-2] + H0[:, Ny-2]))
        #   
        H1[:, Ny-1] = (1.0 - Ce) * H0[:, Ny-1] + Ce * H0[:, Ny-2]
    elif obc_ele[3] == 'Cha_i':
        #
        if CC2.dim()>1: # CC2 is a matrix, ie, grid length is variable
            Ce = CC2         * torch.sqrt(g * (Z[:, Ny-2] + H0[:, Ny-2]))
        else: # CC2 is a number, i.e., grid length is a constant
            Ce = CC2[:,Ny-2] * torch.sqrt(g * (Z[:, Ny-2] + H0[:, Ny-2]))
        #   
        cff2 = 1.0 / (1.0 + Ce)
        H1[:, Ny-1] = cff2 * (H0[:, Ny-1] + Ce * H1[:, Ny-2])
    elif obc_ele[3] == 'Gra':
        H1[:, Ny-1] = H1[:, Ny-2]
    elif obc_ele[3] == 'Clo':
        H1[:, Ny-1] = 0.0
    elif obc_ele[3] == 'Rad':
        if CC2.dim()>1: # CC2 is a matrix, ie, grid length is variable
            Ce = 2.0 * CC2[:,Ny-1] * torch.sqrt(g * (Z[:, Ny-1] + H0[:, Ny-1]))
        else: # CC2 is a number, i.e., grid length is a constant
            Ce = 2.0 * CC2 * torch.sqrt(g * (Z[:, Ny-1] + H0[:, Ny-1]))
        H1[:, Ny-1] = H0[:, Ny-1] - Ce * (H0[:, Ny-1] - H0[:, Ny-2])
    
    #mask_rho = Z > params.dry_limit
    mask_rho =  (H1+Z <= params.MinWaterDepth)
    H1 = torch.where( mask_rho,  H1, 0)
    H = torch.stack((H0,H1),0)
    
    
    return H


def bcond_u2D_vec(H, Z, M, D_M, flux_sign, z_w, z_e, params):
    """
    非就地更新的边界条件函数，返回更新后的 M_new
    参数说明同原函数：
        - H: 水位 (tensor, shape=(2, Nx+1, Ny+1))
        - Z: 水深 (tensor, shape=(Nx+1, Ny+1))
        - M: x方向动量 (tensor, shape=(2, Nx, Ny+1))
        - D_M: 流深 (tensor, shape=(Nx, Ny+1))
        - z_w, z_e: 西、东侧边界高程（列表或tensor）
        - params: 参数结构体，包含 obc_u2d, dt, dx, dy, g, CC1, CC2 等
    """
    eps = 1.0e-6
    obc_u2d = params.obc_u2d
    if H.ndim == 3:
        Nx, Ny = H.shape[1], H.shape[2]
        M_new = M.clone()
    else: # H.ndim==2
        Nx, Ny = H.shape[0], H.shape[1]
        M_new = M.clone()
    
    # ---------------------------
    # 西侧边界更新：更新 M_new[1, 0, :]（更新第0行）
    if obc_u2d[0] == 'Fla':
        #西边界的M，直接去除iy循环即可, LWF
        bry_pgr = -params.g * (H[0, 1] - z_w) * 0.5 * params.CC1[1]
        bry_cor = 0.0 # should add
        cff1 = 1.0 / (0.5 * (Z[0] + H[0, 0] + Z[1] + H[0, 1]))
        bry_str = 0.0 # should add surface forcing
        Cx = 1.0 / np.sqrt(params.g * 0.5 * (Z[0] + H[0, 0] + Z[1] + H[0, 1]))
        cff2 = Cx / params.dx
        bry_val = M[0, 1] / D_M[1] + cff2 * (bry_pgr + bry_cor + bry_str)
        Cx = np.sqrt( params.g * cff1 )
        ubar_w = bry_val - Cx * ( 0.5 * (H[0, 0] + H[0, 1]) - z_w)
        M_new[1, 0] = ubar_w * D_M[0]
        
        # Old loop code
        # ubar_w = torch.zeros((Ny, 1), dtype=float)
        #for iy in range(Ny):
        #     bry_pgr = -params.g * (H[0, 1, iy] - z_w[iy]) * 0.5 * params.CC1
        #     bry_cor = 0.0 # should add
        #     cff1 = 1.0 / (0.5 * (Z[0, iy] + H[0, 0, iy] + Z[1, iy] + H[0, 1, iy]))
        #     bry_str = 0.0 # should add surface forcing
        #     Cx = 1.0 / np.sqrt(params.g * 0.5 * (Z[0, iy] + H[0, 0, iy] + Z[1, iy] + H[0, 1, iy]))
        #     cff2 = Cx / params.dx
        #     bry_val = M[0, 1, iy] / D_M[1, iy] + cff2 * (bry_pgr + bry_cor + bry_str)
        #     Cx = np.sqrt( params.g * cff1 )
        #     ubar_w[iy] = bry_val - Cx * ( 0.5 * (H[0, 0, iy] + H[0, 1, iy]) - z_w[iy])
        #     M[1, 0, iy] = ubar_w[iy] * D_M[0, iy]
    
    elif obc_u2d[0] == 'Gra':
        M_new[1] = F.pad( M_new[1:2, 1:], (0,0,1,0), mode='replicate') #pad for West, up 
    
    elif obc_u2d[0] == 'Clo':
        M_new[1] = F.pad( M_new[1, 1:], (0,0,1,0) ) #pad for West, up 
    
    elif obc_u2d[0] == 'Rad':
        ubar_w = -1.0 * np.sqrt(params.g / (Z[0] + H[1, 0])) * H[1, 0]
        M_new[1, 0] = ubar_w * D_M[0]
    
    # ---------------------------
    # 东侧边界更新：更新 M_new[1, Nx-2, :]（更新第 Nx-2 行）
    #               即 M_new[1, -1]
    if obc_u2d[2] == 'Fla':
        bry_pgr = params.g * (z_e - H[0, -2]) * 0.5 * params.CC1[-2, :]
        bry_cor = 0.0  # should add
        cff1 = 1.0 / (0.5 * (Z[-1] + H[0, -1] + Z[-2] + H[0, -2] + eps))
        bry_str = 0.0  # should add surface forcing
        Cx = 1.0 / np.sqrt(params.g * 0.5 * (Z[-1] + H[0, -1] + Z[-2] + H[0,-2] + eps))
        cff2 = Cx / params.dx
        bry_val = M[0, -2] / D_M[-2] + cff2 * (bry_pgr + bry_cor + bry_str)
        Cx = np.sqrt(params.g * cff1)
        ubar_e = bry_val + Cx * (0.5 * (H[0, -1] + H[0, -2]) - z_e)
        M_new[1, -1] = ubar_e * D_M[-1]
    
    elif obc_u2d[2] == 'Gra':        
        M_new[1] = F.pad( M_new[1:2,:-1], (0,0,0,1), mode='replicate') #pad for East
        
    elif obc_u2d[2] == 'Clo':
        M_new[1] = F.pad( M_new[1,:-1], (0,0,0,1) ) #pad for East
    
    elif obc_u2d[2] == 'Rad':
        ubar_e = 1.0 * np.sqrt(params.g / (Z[-1] + H[1, -1])) * H[1, -1]
        M_new[1, -1] = ubar_e * D_M[-1]
        
    # ---------------------------
    # 南侧边界更新：更新 M_new[1, :, 0] （对 x 方向，更新前 Nx-1 个位置）
    if obc_u2d[1] == 'Fla':
        cff = params.dt * 0.5 / params.dy
        #Du = rho2u(Z[:]+H[0])[:,1]
        cff1 = np.sqrt(params.g * D_M[:, 1])
        #cff1 = np.sqrt(params.g * 0.5 * (Z[ix, 1] + H[0, ix, 1] + Z[ix + 1, 1] + H[0, ix + 1, 1]))
        Ce = cff * cff1
        cff2 = 1.0 / (1.0 + Ce)
        ubar_s = cff2 * (M[0, :, 0] / D_M[:, 0] + Ce * M[1, :, 1] / D_M[:, 1])
        M_new[1, :, 0] = ubar_s * D_M[:, 0]
    
    elif obc_u2d[1] == 'Gra':
        M_new[1] = F.pad( M_new[1:2, :, 1:], (1,0,0,0), mode='replicate') #pad for South, fast by 20%
    
    elif obc_u2d[1] == 'Clo':
        # new_col = torch.zeros((Nx-1,), dtype=M_new.dtype, device=M_new.device).unsqueeze(1)
        # M_new_row = torch.cat([new_col, M_new[1, :, 1:]], dim=1)
        # M_new = torch.stack([M_new[0], M_new_row], dim=0)

        M_new[1] = F.pad( M_new[1, :, 1:], (1,0,0,0) ) #pad for South, fast by 20%
    
    elif obc_u2d[1] == 'Rad':
        cff  = Z[:-1, 0] + H[0, :-1, 0] + Z[1:, 0] + H[0, 1:, 0]
        cff1 = M[0, :, 0] / D_M[:, 0] - M[0, :, 1] / D_M[:, 1]
        cff2 = Z[:-1, 0] + H[1, :-1, 0] + Z[1:, 0] + H[1, 1:, 0] + eps
        CC2_u = rho2u(params.CC2)[:, 0]
        ubar_s = M[0, :, 0] / D_M[:, 0] * cff - 2 * CC2_u * np.sqrt(params.g * cff * 0.5) * cff1
        ubar_s = ubar_s / cff2
        M_new[1, :, 0] = ubar_s * D_M[:, 0]
        
        # cff = Z[ix, 0] + H[0, ix, 0] + Z[ix + 1, 0] + H[0, ix + 1, 0]
        # cff1 = M[0, ix, 0] / D_M[ix, 0] - M[0, ix, 1] / D_M[ix, 1]
        # cff2 = Z[ix, 0] + H[1, ix, 0] + Z[ix + 1, 0] + H[1, ix + 1, 0] + eps
        # ubar_s[ix] = M[0, ix, 0] / D_M[ix, 0] * cff - 2 * params.CC2[ix, 0] * np.sqrt(params.g * cff * 0.5) * cff1
        # ubar_s[ix] = ubar_s[ix] / cff2
        # M[1, ix, 0] = ubar_s[ix] * D_M[ix, 0]
    
    # ---------------------------
    # 北侧边界更新：更新 M_new[1, :, Ny-1]
    if obc_u2d[3] == 'Fla':
        cff = params.dt * 0.5 / params.dy
        cff1 = np.sqrt(params.g * D_M[:,-1]) #see above Southern boundary
        Ce = cff * cff1
        cff2 = 1.0 / (1.0 + Ce)
        ubar_n = cff2 * (M[0, :, -1] / D_M[:, -1] + Ce * M[1, :, -2] / D_M[:, -2])
        M_new[1, :, -1] = ubar_n * D_M[:, -1]
    
    elif obc_u2d[3] == 'Gra':
        M_new[1] = F.pad( M_new[1:2, :, :-1], (0,1,0,0), mode='replicate') #pad for North, fast by 20%

    elif obc_u2d[3] == 'Clo':
        M_new[1] = F.pad( M_new[1, :, :-1], (0,1,0,0) ) #pad for North, fast by 20%
    
    elif obc_u2d[3] == 'Rad':
        cff  = Z[:-1, -1] + H[0, :-1, -1] + Z[1:, -1] + H[0, 1:, -1]
        cff1 = M[0, :, -1] / D_M[:, -1] - M[0, :, -2] / D_M[:, -2] # for M, loop range(Nx-1) means all :
        cff2 = Z[:-1, -1] + H[1, :-1, -1] + Z[1:, -1] + H[1, 1:, -1] + eps
        CC2_u = rho2u(params.CC2)[:, -1]
        ubar_n = M[0, :, -1] / D_M[:, -1] * cff - 2 * CC2_u * np.sqrt(params.g * cff * 0.5) * cff1
        ubar_n = ubar_n / cff2
        M_new[1, :, -1] = ubar_n * D_M[:, -1]
    
    mask_u =  torch.isnan(M_new)
    M_new = torch.where(mask_u,  0, M_new)
    
    return M_new


def bcond_v2D_vec(H, Z, N, D_N, flux_sign, z_s, z_n, params):
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
    eps = 1.0e-6
    obc_v2d = params.obc_v2d
    if H.ndim == 3:
        Nx, Ny = H.shape[1], H.shape[2]  # 注意：N 的空间尺寸一般为 (Nx+1, Ny)
        N_new = N.clone()
    else: # H.ndim==2
        Nx, Ny = H.shape[0], H.shape[1]
        N_new = N.clone()
    
    # ---------------------------
    # 西侧边界更新 —— 对应 N_new[1, 0, :] （横向第 0 行）
    if obc_v2d[0] == 'Fla':
        cff = params.dt * 0.5 / params.dx
        cff1 = np.sqrt(params.g * 0.5 * (Z[1,:-1] + H[0, 1,:-1] + Z[1, 1:] + H[0, 1, 1:]))
        Cx = cff * cff1
        cff2 = 1.0 / (1.0 + Cx)
        vbar_w = cff2 * (N[0, 0] / D_N[0] + Cx * N[1, 1] / D_N[1])
        N_new[1, 0] = vbar_w * D_N[0]
    elif obc_v2d[0] == 'Gra':
        N_new[1] = F.pad( N_new[1:2, 1:], (0,0,1,0), mode='replicate') #pad for West
        
    elif obc_v2d[0] == 'Clo':
        N_new[1] = F.pad( N_new[1, 1:], (0,0,1,0) ) #pad for West
        
    elif obc_v2d[0] == 'Rad':
        cff = Z[0, :-1] + H[0, 0, :-1] + Z[0, 1:] + H[0, 0, 1:]
        cff1 = N[0, 0] / D_N[0] - N[0, 1] / D_N[1]
        cff2 = Z[0,:-1] + H[1, 0, :-1] + Z[0, 1:] + H[1, 0, 1:] + eps
        CC1_v = rho2v(params.CC1)[0]
        vbar_w = N[0, 0] / D_N[0] * cff - 2 * CC1_v * np.sqrt(params.g * cff * 0.5) * cff1
        vbar_w = vbar_w  / cff2
        N_new[1, 0] = vbar_w * D_N[0]
       
    # ---------------------------
    # Eastern side更新 —— 对应 N_new[1, Nx, :]（最后一行，东侧）
    if obc_v2d[2] == 'Fla':
        cff = params.dt * 0.5 / params.dx
        cff1 = np.sqrt(params.g * 0.5 * (Z[-2, :-1] + H[0, -2, :-1] + Z[-2, 1:] + H[0, -2, 1:]) + eps)
        Cx = cff * cff1
        cff2 = 1.0 / (1.0 + Cx)
        vbar_e = cff2 * (N[0, Nx - 1] / D_N[Nx - 1] + Cx * N[1, Nx - 2] / D_N[Nx - 2])
        N_new[1, Nx - 1] = vbar_e * D_N[Nx - 1]
    elif obc_v2d[2] == 'Gra':
        N_new[1] = F.pad( N_new[1:2, :-1], (0,0,0,1), mode='replicate') #pad for East
        
    elif obc_v2d[2] == 'Clo':
        N_new[1] = F.pad( N_new[1, :-1], (0,0,0,1) ) #pad for East
        
    elif obc_v2d[2] == 'Rad':
        cff = Z[Nx - 1, :-1] + H[0, Nx - 1,:-1] + Z[Nx - 1, 1:] + H[0, Nx - 1, 1:]
        cff1 = N[0, Nx - 1] / D_N[Nx - 1] - N[0, Nx - 2] / D_N[Nx - 2]
        cff2 = Z[Nx - 1,:-1] + H[1, Nx - 1,:-1] + Z[Nx - 1, 1:] + H[1, Nx - 1, 1:]
        CC1_v = rho2v(params.CC1)[Nx-1]
        vbar_e = N[0, Nx - 1] / D_N[Nx - 1] * cff - 2 * CC1_v * np.sqrt(params.g * cff * 0.5) * cff1
        vbar_e = vbar_e / cff2
        N_new[1, Nx - 1] = vbar_e * D_N[Nx - 1]
    
    # ---------------------------
    # 南侧边界更新 —— 对应 N_new[1, :, 0]（竖向第 0 列）
    if obc_v2d[1] == 'Fla':
        #for ix in range(Nx):
        #bry_pgr = -params.g * (H[0, :, 0] - z_s) * 0.5 * params.CC2[:, 0] ##Liu 
        #should be -params.g * (H[0, :, 2] - z_s) * 0.5 * params.CC2[:, 0] 
        bry_pgr = -params.g * (H[0, :, 2] - z_s) * 0.5 * params.CC2[:, 0]
        bry_cor = 0.0
        cff1 = 1.0 / (0.5 * (Z[:, 0] + H[0, :, 0] + Z[:, 1] + H[0, :, 1]))
        bry_str = 0.0
        Ce = 1.0 / np.sqrt(params.g * 0.5 * (Z[:, 0] + H[0, :, 0] + Z[:, 1] + H[0, :, 1]))
        cff2 = Ce / params.dy
        bry_val = N[0, :, 1] / D_N[:, 1] + cff2 * (bry_pgr + bry_cor + bry_str)
        Ce = np.sqrt(params.g * cff1)
        vbar_s = bry_val - Ce * (0.5 * (H[0, :, 0] + H[0, :, 1]) - z_s)
        N_new[1, :, 0] = vbar_s * D_N[:, 0]
        
    elif obc_v2d[1] == 'Gra':
        N_new[1] = F.pad( N_new[1:2, :, 1:], (1,0,0,0), mode='replicate') #pad for South, fast by 20%
        
    elif obc_v2d[1] == 'Clo':
        N_new[1] = F.pad( N_new[1, :, 1:], (1,0,0,0) ) #pad for South, fast by 20%
        
    elif obc_v2d[1] == 'Rad':
        vbar_s = -1.0 * np.sqrt(params.g / (Z[:, 0] + H[1, :, 0] + eps)) * H[1, :, 0]
        N_new[1, :, 0] = vbar_s * D_N[:, 0]
        
    # ---------------------------
    # Northern side更新 —— 对应 N_new[1, :, Ny-2]
    if obc_v2d[3] == 'Fla':
        bry_pgr = -params.g * (z_n - H[0, :, Ny-3]) * 0.5 * params.CC2[:,Ny-3]
        bry_cor = 0.0 # should add
        cff1 = 1.0 / (0.5 * (Z[:, Ny - 3] + H[0, :, Ny - 3] + Z[:, Ny - 2] + H[0, :, Ny - 2] + eps))
        #TODO(Liu) here Ny-3, Ny-2 or Ny-2,Ny-1  ?
        bry_str = 0.0  # should add surface forcing
        Ce = 1.0 / np.sqrt(params.g * 0.5 * (Z[:, Ny - 2] + H[0, :, Ny - 2] + Z[:, Ny - 3] + H[0, :, Ny - 3] + eps))
        cff2 = Ce / params.dy
        bry_val = N[0, :, Ny - 3] / D_N[:, Ny - 3] + cff2 * (bry_pgr + bry_cor + bry_str)
        Ce = np.sqrt(params.g * cff1)
        vbar_n = bry_val + Ce * (0.5 * (H[0, :, Ny - 3] + H[0, :, Ny - 2]) - z_n)
        N_new[1, :, Ny - 2] = vbar_n * D_N[:, Ny-2]
        
    elif obc_v2d[3] == 'Gra':
        N_new[1] = F.pad( N_new[1:2, :, :-1], (0,1,0,0), mode='replicate') #pad for North, fast by 20%
        
    elif obc_v2d[3] == 'Clo':
        N_new[1] = F.pad( N_new[1, :, :-1], (0,1,0,0) ) #pad for North, fast by 20%
        
    elif obc_v2d[3] == 'Rad':
        vbar_n = 1.0 * np.sqrt(params.g / (Z[:, Ny - 1] + H[1, :, Ny - 1] + eps)) * H[1, :, Ny - 1]
        N_new[1, :, Ny - 2] = vbar_n * D_N[:, Ny - 2]
    
    mask_v =  torch.isnan(N_new)
    N_new = torch.where(mask_v,  0, N_new)
    
    return N_new
