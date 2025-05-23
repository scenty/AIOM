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
from tool_train import ddx,ddy,rho2u,rho2v,v2rho,u2rho,dd
from tool_train import ududx_up,vdudy_up,udvdx_up,vdvdy_up,tic,toc
from grid import Grid
from tidePred import compute_pt_tide_ts
import datetime
import pandas as pd
from dynamics_core import bcond_zeta_vec,bcond_u2D_vec,bcond_v2D_vec

class Params:
    def __init__(self):
        # Domain parameters
        # self.Lx = 800*1e3
        # self.Ly = 400*1e3
        # self.Nx = 100
        # self.Ny = 50
        # self.dx = self.Lx / self.Nx
        # self.dy = self.Ly / self.Ny
        # self.depth = 50.0
        self.lon0 = 111
        self.lon1 = 127
        self.lat0 = 16
        self.lat1 = 33
        self.dx = 5000
        self.dy = 5000
        # lon0, lat0, lon1, lat1
        self.grid = Grid(self.lon0, self.lat0, self.lon1, self.lat1, self.dx, self.dy)
        self.t_date_0 = datetime.datetime(year=2022, month=1, day=1)
        self.t_date_1 = datetime.datetime(year=2022, month=1, day=15)
        self.bc_timestep = 1800
        # Physical constants
        self.g = 9.8
        self.rho_water = 1025.0
        self.rho_air = 1.2
        self.Cd = 2.5e-3
        self.manning = 3.0e-3
        self.dry_limit = 0.2
        self.MinWaterDepth = 0.01
        self.FrictionDepthLimit = 5e-3
        self.f_cor = 0.0 #1e-5
        
        # Time parameters
        self.dt = 80        
        self.NT = 1000
        self.cal_unit = 600
        self.CFL = 0.25
        self.eps_time = 1.0e-7
        self.centerWeighting0 = 0.9998
        
        # Boundary conditions
        #                 W      S      E      N
        self.obc_ele = ['Clo', 'Rad', 'Cha_i', 'Rad']
        self.obc_u2d = ['Clo', 'Rad', 'Fla', 'Rad']
        self.obc_v2d = ['Clo', 'Rad', 'Fla', 'Rad']
        
        # Temporary variables
        self.CC1 = torch.from_numpy(self.dt * self.grid.pm) #self.dt / self.dx
        self.CC2 = torch.from_numpy(self.dt * self.grid.pm) #self.dt / self.dy
        self.CC3 = self.CC1 * self.g
        self.CC4 = self.CC2 * self.g
        
        # Wind parameters
        self.Wr = 30
        self.rMax = 50*1000
        self.typhoon_Vec = [5,0] #m/s
        self.typhoon_Pos = [[200*1e3 + itime*self.dt*self.typhoon_Vec[0], 
                            200*1e3 + itime*self.dt*self.typhoon_Vec[1]]
                            for itime in range(self.NT)]

def cal_dynamic_timeStep(params, H, U, V):
    mintimestep = 300
    condition = (H + params.grid.h) < params.MinWaterDepth
    min_water_depth = torch.zeros_like(H).float() + params.MinWaterDepth
    depth_g = torch.where(condition, min_water_depth, (H + params.grid.h).float()) * params.g

    term0 = (depth_g.sqrt() + abs(F.pad(U, [0, 0, 0, 1]))) / params.grid.pn
    term1 = (depth_g.sqrt() + abs(F.pad(V, [0, 1, 0, 0]))) / params.grid.pm
    criticalDt = params.CFL / (params.grid.pm * params.grid.pn) / (term0 + term1)
    mask_z = (params.grid.h <= -params.dry_limit)
    criticalDt = torch.where(torch.from_numpy(mask_z) , torch.zeros_like(criticalDt)+mintimestep, criticalDt)
    dt = min(criticalDt.min(), mintimestep)
    return dt

def interp_bc_modelTime(params, ttPred, tidePred):
    ttModel = [params.t_date_0 + datetime.timedelta(seconds = params.dt * t)
                  for t in range(params.NT)]
    nsta = tidePred.shape[0]
    tideModel = np.zeros((nsta, params.NT))
    for ista in range(nsta):
        _ele = tidePred[ista, :]
        s = pd.Series(_ele, index=ttPred)
        # 直接插值 + 边缘填充
        s_interpolated = (
            s.reindex(ttModel)
                .interpolate(method='time')
                .ffill()  # 处理早于原始时间的点
                .bfill()  # 处理晚于原始时间的点
        )
        # 提取结果
        tideModel[ista, :] = s_interpolated.loc[ttModel]
    return tideModel

def get_bd_Pred(params):
    tide_model_path = r".\DATA\tpxo8_atlas"
    # for east
    lon_e = params.grid.bound['east'][1]
    lat_e = params.grid.bound['east'][2]
    tide_e_ = compute_pt_tide_ts((lon_e, lat_e), params.t_date_0, params.t_date_1, tide_model_path, params.bc_timestep)
    # tide_e_z = interp_bc_modelTime(params, tide_e_['tide_time'], tide_e_['tide_z'])
    # tide_e_u = interp_bc_modelTime(params, tide_e_['tide_time'], tide_e_['tide_u'])
    # tide_e_v = interp_bc_modelTime(params, tide_e_['tide_time'], tide_e_['tide_v'])
    # tide_e = {'tide_z': tide_e_z,
    #         'tide_u': tide_e_u,
    #         'tide_v': tide_e_v}

    # for west
    lon_w = params.grid.bound['west'][1]
    lat_w = params.grid.bound['west'][2]
    tide_w_ = compute_pt_tide_ts((lon_w, lat_w), params.t_date_0, params.t_date_1, tide_model_path, params.bc_timestep)
    # tide_w_z = interp_bc_modelTime(params, tide_w_['tide_time'], tide_w_['tide_z'])
    # tide_w_u = interp_bc_modelTime(params, tide_w_['tide_time'], tide_w_['tide_u'])
    # tide_w_v = interp_bc_modelTime(params, tide_w_['tide_time'], tide_w_['tide_v'])
    # tide_w = {'tide_z': tide_w_z,
    #           'tide_u': tide_w_u,
    #           'tide_v': tide_w_v}
    # for north
    lon_n = params.grid.bound['north'][1]
    lat_n = params.grid.bound['north'][2]
    tide_n_ = compute_pt_tide_ts((lon_n, lat_n), params.t_date_0, params.t_date_1, tide_model_path, params.bc_timestep)
    # tide_n_z = interp_bc_modelTime(params, tide_n_['tide_time'], tide_n_['tide_z'])
    # tide_n_u = interp_bc_modelTime(params, tide_n_['tide_time'], tide_n_['tide_u'])
    # tide_n_v = interp_bc_modelTime(params, tide_n_['tide_time'], tide_n_['tide_v'])
    # tide_n = {'tide_z': tide_n_z,
    #           'tide_u': tide_n_u,
    #           'tide_v': tide_n_v}
    # for south
    lon_s = params.grid.bound['south'][1]
    lat_s = params.grid.bound['south'][2]
    tide_s_ = compute_pt_tide_ts((lon_s, lat_s), params.t_date_0, params.t_date_1, tide_model_path, params.bc_timestep)
    # tide_s_z = interp_bc_modelTime(params, tide_s_['tide_time'], tide_s_['tide_z'])
    # tide_s_u = interp_bc_modelTime(params, tide_s_['tide_time'], tide_s_['tide_u'])
    # tide_s_v = interp_bc_modelTime(params, tide_s_['tide_time'], tide_s_['tide_v'])
    # tide_s = {'tide_z': tide_s_z,
    #           'tide_u': tide_s_u,
    #           'tide_v': tide_s_v}
    
    return tide_e_, tide_w_, tide_n_, tide_s_


def mass_cartesian_torch(H, Z, M, N, params, tide_z):
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
    H1[1:-1,1:-1] = H0[1:-1,1:-1] - CC1[1:-1,1:-1] * dMdx[1:-1,1:-1] - CC2[1:-1,1:-1] * dNdy[1:-1,1:-1]
        
    #TODO    
    # 干湿修正（使用 torch.where 保证全为新张量）
    mask_deep = (Z <= -dry_limit)
    H1 = torch.where(mask_deep, torch.zeros_like(H1), H1)

    ZH0 = Z + H0
    ZH1 = Z + H1
    #可能会产生干湿的所有单元，其地形在+-dry_limit之间
    cond = (Z < dry_limit) & (Z > -dry_limit) 

    wet_to_dry = cond & (ZH0>0) & ((H1-H0)<0) & (ZH1<=MinWaterDepth)
    c1 = wet_to_dry & (Z>0)
    c2 = wet_to_dry & (Z<=0)
    H1 = torch.where(c1, -Z + MinWaterDepth, H1) # liuy add MinWaterDepth
    H1 = torch.where(c2, torch.zeros_like(H1), H1)

    cond_dry = cond & (ZH0<=0)
    c3 = cond_dry & ((H1-H0)>0)
    H1 = torch.where(c3, H1 - H0 - Z, H1) # dry to wet
    c4 = cond_dry & ((H1-H0)<=0) & (Z>0)  # dry to dry
    c5 = cond_dry & ((H1-H0)<=0) & (Z<=0) # dry to dry
    H1 = torch.where(c4, -Z + MinWaterDepth, H1) # liuy add MinWaterDepth
    H1 = torch.where(c5, torch.zeros_like(H1), H1)
       
    # 构造新的 H，不再对 H.clone() 进行切片赋值，而是用 stack 构造新张量
    H_new = torch.stack((H0, H1), dim=0)
    H_new = bcond_zeta_vec(H_new, Z, params, tide_z)
   
    # tic()
    # H_new = bcond_zeta(H_new, Z, params, tide_z)
    # toc()
    # assert not torch.any((H_vec-H_new) > 1e-6)
    
    assert not torch.any(torch.isnan(H_new))
    #TODO boundary tide
    #H_new[1,1,:] = 1.0 * np.sin(2 * np.pi / 200 * params.itime * params.dt)
    return H_new

def momentum_nonlinear_cartesian_torch(H, Z, M, N, Wx, Wy, Pa, params, tide_z, tide_u, tide_v, manning):
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
    Nx = params.grid.Nx
    Ny = params.grid.Ny
        
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
    Pre_grad_x = CC1[0:-1, :] * D0 * dPdx / rho_water
    
    ududx = F.pad( ududx_up(M[0],N[0],Z+H[1]), (0,0,1,1)) #pad for up and down
    vdudy = F.pad( vdudy_up(M[0],N[0],Z+H[1]), (1,1,0,0)) #pad for left and right
    
    # Nu is applied here and in the friction below
    N_exp = torch.cat( (N[0,:,0:1], N[0], N[0,:,-1:]), dim = 1)
    Nu = rho2u(v2rho(N_exp))   #at u-point
    # Nu = F.pad( rho2u(rho2v(N[0])) , (1,1))
    # Torch
    phi_M = 1.0 - CC1[0:-1, :] * torch.min(torch.abs(m0 / torch.clamp(D0, min=MinWaterDepth)), torch.sqrt(g*D0))
    M_val = (phi_M*m0 + 0.5*(1.0 - phi_M)*(m1 + m2)        #implicit friction
            + dt  * ( sustr + f_cor * Nu)                  #wind stress and coriolis
            - CC1[0:-1, :] * ududx                                  #u advection
            - CC2[0:-1, :] * vdudy                                  #v advection
            - CC3[0:-1, :] * D0*(h2u_M - h1u_M) + Pre_grad_x)         #pgf
    
    #pcolor(rho2u(X),rho2u(Y),ududx);colorbar();show()
    #pcolor(rho2u(X),rho2u(Y),vdudy);colorbar();show()

    # 底摩擦
    friction_mask = (D0 > FrictionDepthLimit)    
    #Nu was generated before
    Fx = g * Cf[0:-1,:]**2 / (D0**2.33 + 1e-9) * torch.sqrt(m0**2 + Nu**2) * m0
    Fx = torch.where(torch.abs(dt*Fx)>torch.abs(m0), m0/dt, Fx)
    Fx = torch.where(friction_mask, Fx, torch.zeros_like(Fx))
    M_val = M_val - dt * Fx
  
    mask_fs999_M = (M_val * flux_sign_M < 0) & (flux_sign_M != 0)
    M_val = torch.where(mask_fs999_M, torch.zeros_like(M_val), M_val)
    
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
    Pre_grad_y = CC2[:, :-1] * D0N * dPdy / rho_water
    
    udvdx = F.pad( udvdx_up(M[0],N[0],Z+H[1]), (0,0,1,1)) #pad for up and down
    vdvdy = F.pad( vdvdy_up(M[0],N[0],Z+H[1]), (1,1,0,0)) #pad for left and right
    
    # Mv is applied here and in the friction below
    M_exp = torch.cat( (M[0,0:1],M[0], M[0,-1:]), dim =0)
    Mv = rho2u(v2rho(M_exp)) #at v-point, [ 1 ~ Nx-1, 1 ~ Ny-1]
    #Mv = F.pad( rho2u(rho2v( M[0])) , (0,0,1,1)) #pad for up and down
    phi_N = 1.0 - CC2[:, :-1] * torch.min(torch.abs(n0 / torch.clamp(D0N, min=MinWaterDepth)), torch.sqrt(g*D0N))
    N_val = (phi_N*n0 + 0.5*(1.0 - phi_N)*(n1 + n2) 
            + dt * (svstr - f_cor * Mv )
            - CC1[:, :-1] * udvdx
            - CC2[:, :-1] * vdvdy
            - (CC4[:, :-1] * D0N * (h2u_N - h1u_N) + Pre_grad_y))

    #底摩擦
    friction_maskN = (D0N > FrictionDepthLimit)
    Fy = g * Cf[:, 0:-1]**2 / (D0N**2.33 + 1e-9) * torch.sqrt(Mv**2 + n0**2) * n0
    Fy = torch.where(torch.abs(dt*Fy)>torch.abs(n0), n0/dt, Fy)
    Fy = torch.where(friction_maskN, Fy, torch.zeros_like(Fy))
    N_val = N_val - dt * Fy
    
    mask_fs999_N = (N_val * flux_sign_N < 0) & (flux_sign_N != 0)
    
    N_val = torch.where(mask_fs999_N, torch.zeros_like(N_val), N_val)
    
    N_new = N.clone()
    N_new[1,1:-1,1:-1] = N_val[1:-1,1:-1]
    #pcolor(rho2v(X)[1:-1,1:-1],rho2v(Y)[1:-1,1:-1],N_val[1:-1,1:-1]/D_N[1:-1,1:-1]);colorbar();show()
    
    # apply boundary condition
    z_w = torch.tensor(0) #tide_z['west'] #torch.from_numpy(0.5 * np.sin(2 * np.pi / 5 * itime * dt) * (np.zeros((Ny + 1, 1)) + 1))
    z_e = torch.from_numpy(tide_z['east'])
    z_n = torch.tensor(0)
    z_s = torch.tensor(0)
    #TODO check 
    M_new = bcond_u2D_vec(H, Z, M_new, D_M, flux_sign_M, z_w, z_e, params)
    N_new = bcond_v2D_vec(H, Z, N_new, D_N, flux_sign_N, z_s, z_n, params)
    
    if torch.any(torch.isnan(M_new)):
        print(torch.where(torch.isnan(M_new)))
    assert not torch.any(torch.isnan(M_new))
    if torch.any(torch.isnan(N_new)):
        print(torch.where(torch.isnan(N_new)))
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
    # ! This subroutine checks the reasonble situations of flux M/N
    # ! FluxSign can be {-2,-1,0,1,2,999}, with the same definition as Sign_M/N in LayerParameters
    # ! dryLimit      : permanent dry limit defined in GlobalParametes
    # ! z1,z2,h1,h2   : water depth and elevation on both sides of flux M/N
    # ! 0 :water can flow in both directions; 999:water cannot flow
    # ! 1 :water can only flow in positive direction, and from higher to lower
    # ! 2 :water can only flow in positive direction, and from lower to higher
    # !-1 :water can only flow in negative direction, and from higher to lower
    # !-2 :water can only flow in negative direction, and from lower to higher
    # ! If water can flow in only one direction, values of h1 and h2 are modified for momentum equation    
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


def bcond_zeta(H, Z, params, tide_z):
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
            Cx = CC1[1, iy] * np.sqrt(g * ( Z[1, iy] + H[0, 1, iy]))
            H[1, 0, iy] = (1.0 - Cx) * H[0, 0, iy] + Cx * H[0, 1, iy]
    elif obc_ele[0] == 'Cha_i': # implicit Chapman boundary condition.
        for iy in range(Ny):
            Cx = CC1[1, iy] * np.sqrt(g * ( Z[1, iy] + H[0, 1, iy]))
            cff2 = 1.0 / (1.0 + Cx)
            H[1, 0, iy] = cff2 * (H[0, 0, iy] + Cx * H[1, 1, iy])
    elif obc_ele[0] == 'Gra':
        H[1, 1, :] = H[1, 2, :]
    elif obc_ele[0] == 'Clo':
        H[1, 0, :] = 0.0
    elif obc_ele[0] == 'Rad':
        for iy in range(Ny):
            # Only water Point
            if Z[0, iy] > params.dry_limit:
                H[1, 0, iy] = H[0, 0, iy] - 2 * CC1[0, iy] / np.sqrt(g * (Z[0, iy] + H[0, 0, iy])) * (H[0, 0, iy] - H[0, 1, iy]) 
    # eastern side
    if obc_ele[2] == 'Cha_e': # explicit Chapman boundary condition
        for iy in range(Ny):
            Cx = CC1[Nx - 2, iy] * np.sqrt(g * (Z[Nx - 2, iy] + H[0, Nx - 2, iy]))
            H[1, Nx - 1, iy] = (1. - Cx) * H[0, Nx - 1, iy] + Cx * H[0, Nx - 2, iy]
    elif obc_ele[2] == 'Cha_i':  # implicit Chapman boundary condition.
        for iy in range(Ny):
            Cx = CC1[Nx - 2, iy] * np.sqrt(g * (Z[Nx - 2, iy] + H[0, Nx - 2, iy]))
            cff2 = 1. / (1. + Cx)
            H[1, Nx - 1, iy] = cff2 * (H[0, Nx - 1, iy] + Cx * H[1, Nx - 2, iy])
    elif obc_ele[2] == 'Gra':
        H[1, Nx - 1, :] = H[1, Nx - 2, :]
    elif obc_ele[2] == 'Clo':
        H[1, Nx - 1, :] = 0.0
    elif obc_ele[2] == 'Rad':
        for iy in range(Ny):
            # Only water Point
            if Z[Nx - 1, iy] > 0.0: 
                H[1, Nx - 1, iy] = H[0, Nx - 1, iy] - 2 * CC1[Nx - 1, iy] / np.sqrt(g * (Z[Nx - 1, iy] + H[0, Nx - 1, iy])) * (H[0, Nx - 1, iy] - H[0, Nx - 2, iy])

    # southern side
    if obc_ele[1] == 'Cha_e': # explicit Chapman boundary condition
        for ix in range(Nx):
            Ce = CC2[ix, 1] * np.sqrt(g * (Z[ix, 1] + H[0, ix, 1]))
            H[1, ix, 0] = (1. - Ce) * H[0, ix, 0] + Ce * H[0, ix, 1]
    elif obc_ele[1] == 'Cha_i':  # implicit Chapman boundary condition.
        for ix in range(Nx):
            Ce = CC2[ix, 1] * np.sqrt(g * (Z[ix, 1] + H[0, ix, 1]))
            cff2 = 1. / (1. + Ce)
            H[1, ix, 0] = cff2 * (H[0, ix, 0] + Ce * H[1, ix, 1])
    elif obc_ele[1] == 'Gra':
        H[1, :, 0] = H[1, :, 1]
    elif obc_ele[1] == 'Clo':
        H[1, :, 0] = 0.0
    elif obc_ele[1] == 'Rad':
        for ix in range(Nx):
            # Only water Point
            if Z[ix, 0] > params.dry_limit: 
                H[1, ix, 0] = H[0, ix, 0] - 2. * CC2[ix, 0] * np.sqrt(g * (Z[ix, 0] + H[0, ix, 0])) * (H[0, ix, 0] - H[0, ix, 1])
    # northern side
    if obc_ele[3] == 'Cha_e': # explicit Chapman boundary condition
        for ix in range(Nx):
            Ce = CC2[ix, Ny - 2] * np.sqrt(g * (Z[ix, Ny - 2] + H[0, ix, Ny - 2]))
            H[1, ix, Ny - 1] = (1. - Ce) * H[0, ix, Ny - 1] + Ce * H[0, ix, Ny - 2]
    elif obc_ele[3] == 'Cha_i':  # implicit Chapman boundary condition.
        for ix in range(Nx):
            Ce = CC2[ix, Ny - 2] * np.sqrt(g * (Z[ix, Ny - 2] + H[0, ix, Ny - 2]))
            cff2 = 1. / (1. + Ce)
            H[1, ix, Ny - 1] = cff2 * (H[0, ix, Ny - 1] + Ce * H[1, ix, Ny-2])
    elif obc_ele[3] == 'Gra':
        H[1, :, Ny - 1] = H[1, :, Ny - 2]
    elif obc_ele[3] == 'Clo':
        H[1, :, Ny - 1] = 0.0
    elif obc_ele[3] == 'Rad':
        for ix in range(Nx):
            # Only water Point
            if Z[ix, Ny - 1] > params.dry_limit: 
                H[1, ix, Ny - 1] = H[0, ix, Ny - 1] - 2. * CC2[ix, Ny - 1] * np.sqrt(g * (Z[ix, Ny - 1] + H[0, ix, Ny - 1])) * (H[0, ix, Ny - 1] - H[0, ix, Ny - 2])
    return H

def bcond_u2D(H, Z, M, D_M, flux_sign, z_w, z_e, params):
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
    eps = 1.0e-6
    obc_u2d = params.obc_u2d
    Nx, Ny = H.shape[1], H.shape[2]  # 此处： L = Nx, M = Ny, Lm = Nx - 1, Mm = Ny - 1
    # Southern side
    if obc_u2d[1] == 'Fla':
        ubar_s = torch.zeros((Nx - 1, 1), dtype=float)
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
        ubar_s = torch.zeros((Nx - 1, 1), dtype=float)
        for ix in range(Nx-1):
            if (flux_sign[ix, 0] != 999) & (flux_sign[ix, 1] != 999):
                cff = Z[ix, 0] + H[0, ix, 0] + Z[ix + 1, 0] + H[0, ix + 1, 0]
                cff1 = M[0, ix, 0] / D_M[ix, 0] - M[0, ix, 1] / D_M[ix, 1]
                cff2 = Z[ix, 0] + H[1, ix, 0] + Z[ix + 1, 0] + H[1, ix + 1, 0] + eps
                ubar_s[ix] = M[0, ix, 0] / D_M[ix, 0] * cff - 2 * params.CC2[ix, 0] * np.sqrt(params.g * cff * 0.5) * cff1
                ubar_s[ix] = ubar_s[ix] / cff2
                M[1, ix, 0] = ubar_s[ix] * D_M[ix, 0]
    # northern side
    if obc_u2d[3] == 'Fla':
        ubar_n = torch.zeros((Nx - 1, 1), dtype=float)
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
        ubar_s = torch.zeros((Nx - 1, 1), dtype=float)
        for ix in range(Nx-1):
            if (flux_sign[ix, Ny - 1] !=999) & (flux_sign[ix, Ny - 2] !=999):
                cff = Z[ix, Ny - 1] + H[0, ix, Ny - 1] + Z[ix + 1, Ny - 1] + H[0, ix + 1, Ny - 1]
                cff1 = M[0, ix, Ny - 1] / D_M[ix, Ny - 1] - M[0, ix, Ny - 2] / D_M[ix, Ny - 2]
                cff2 = Z[ix, Ny - 1] + H[1, ix, Ny - 1] + Z[ix + 1, Ny - 1] + H[1, ix + 1, Ny - 1] + eps
                ubar_s[ix] = M[0, ix, Ny - 1] / D_M[ix, Ny - 1] * cff - 2 * params.CC2[ix, Ny - 1] * np.sqrt(params.g * cff * 0.5) * cff1
                ubar_s[ix] = ubar_s[ix] / cff2
                M[1, ix, Ny - 1] = ubar_s[ix] * D_M[ix, Ny - 1]
    # western side
    if obc_u2d[0] == 'Fla':
        ubar_w = torch.zeros((Ny, 1), dtype=float)
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
            M[1, 0, iy] = ubar_w[iy] * D_M[0, iy]

    elif obc_u2d[0] == 'Gra':
        M[1, 0, :] = M[1, 1, :]
    elif obc_u2d[0] == 'Clo':
        M[1, 0, :] = 0.0
    elif obc_u2d[0] == 'Rad':
        ubar_w = torch.zeros((Ny, 1), dtype=float)
        for iy in range(Ny):
            if Z[0, iy] > params.dry_limit:
                ubar_w[iy] = -1.0 * np.sqrt(params.g / (Z[0, iy] + H[1, 0, iy])) * H[1, 0, iy]
                M[1, 0, iy] = ubar_w[iy] * D_M[0, iy]
    # Eastern side
    if obc_u2d[2] == 'Fla':
        ubar_e = torch.zeros((Ny, 1), dtype=float)
        for iy in range(Ny):
            #if Z[Nx - 1, iy] > 0.0 : #& D_M[Nx - 3, iy] != 0:
            if flux_sign[Nx - 3, iy] != 999:
                bry_pgr = params.g * (z_e[iy] - H[0, Nx - 3, iy]) * 0.5 * params.CC1[Nx - 3, iy]
                bry_cor = 0.0  # should add
                cff1 = 1.0 / (0.5 * (Z[Nx - 3, iy] + H[0, Nx - 3, iy] + Z[Nx - 2, iy] + H[0, Nx - 2, iy] + eps))
                bry_str = 0.0  # should add surface forcing
                Cx = 1.0 / np.sqrt(params.g * 0.5 * (Z[Nx - 2, iy] + H[0, Nx - 2, iy] + Z[Nx - 3, iy] + H[0, Nx - 3, iy] + eps))
                cff2 = Cx / params.dx
                bry_val = M[0, Nx - 3, iy] / D_M[Nx - 3, iy] + cff2 * (bry_pgr + bry_cor + bry_str)
                Cx = np.sqrt(params.g * cff1)
                ubar_e[iy] = bry_val + Cx * (0.5 * (H[0, Nx - 3, iy] + H[0, Nx - 2, iy]) - z_e[iy])
                M[1, Nx - 2, iy] = ubar_e[iy] * D_M[Nx - 2, iy]
        #print("Finish Fla east")
    elif obc_u2d[2] == 'Gra':
        M[1, Nx - 2, :] = M[1, Nx - 3, :]
    elif obc_u2d[2] == 'Clo':
        M[1, Nx - 2, :] = 0.0
    elif obc_u2d[2] == 'Rad':
        ubar_e = torch.zeros((Ny, 1), dtype=float)
        for iy in range(Ny):
            if Z[Nx - 1, iy] > params.dry_limit:
                ubar_e[iy] = 1.0 * np.sqrt(params.g / (Z[Nx - 1, iy] + H[1, Nx - 1, iy])) * H[1, Nx - 1, iy]
                M[1, Nx - 2, iy] = ubar_e[iy] * D_M[Nx - 2, iy]

    return M

def bcond_v2D(H, Z, N, D_N, flux_sign, z_s, z_n, Params):
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
    Nx, Ny = H.shape[1], H.shape[2]
    
    # western side
    if obc_v2d[0] == 'Fla':
        vbar_w = torch.zeros((Ny - 1, 1), dtype=float)
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
        vbar_w = torch.zeros((Ny - 1, 1), dtype=float)
        for iy in range(Ny - 1):
            #if Z[0, iy] > params.dry_limit & Z[0, iy + 1] > params.dry_limit & flux_sign[0, iy] !=999 &:
            #if flux_sign[0, iy] !=999 & flux_sign[1, iy] !=999:
            if notequal(flux_sign[0, iy], 999) & notequal(flux_sign[0, iy], 999):
                cff = Z[0, iy] + H[0, 0, iy] + Z[0, iy + 1] + H[0, 0, iy + 1]
                cff1 = N[0, 0, iy] / D_N[0, iy] - N[0, 1, iy] / D_N[1, iy]
                cff2 = Z[0, iy] + H[1, 0, iy] + Z[0, iy + 1] + H[1, 0, iy + 1] + eps
                vbar_w[iy] = N[0, 0, iy] / D_N[0, iy] * cff - 2 * params.CC1[0, iy] * np.sqrt(params.g * cff * 0.5) * cff1
                vbar_w[iy] = vbar_w[iy] / cff2
                N[1, 0, iy] = vbar_w[iy] * D_N[0, iy]
    
    # Eastern side
    if obc_v2d[2] == 'Fla':
        vbar_e = torch.zeros((Ny - 1, 1), dtype=float)
        for iy in range(Ny - 1):
            if (flux_sign[Nx - 1, iy] !=999) & (flux_sign[Nx - 2, iy] !=999):
                cff = params.dt * 0.5 / params.dx
                cff1 = np.sqrt(params.g * 0.5 * (Z[Nx - 2, iy] + H[0, Nx - 2, iy] + Z[Nx - 2, iy + 1] + H[0, Nx - 2, iy + 1]) + eps)
                Cx = cff * cff1
                cff2 = 1.0 / (1.0 + Cx)
                vbar_e[iy] = cff2 * (N[0, Nx - 1, iy] / D_N[Nx - 1, iy] + Cx * N[1, Nx - 2, iy] / D_N[Nx - 2, iy])
                N[1, Nx - 1, iy] = vbar_e[iy] * D_N[Nx - 1, iy]
    elif obc_v2d[2] == 'Gra':
        N[1, Nx - 1, :] = N[1, Nx - 2, :]
    elif obc_v2d[2] == 'Clo':
        N[1, Nx - 1, :] = 0.0
    elif obc_v2d[2] == 'Rad':
        vbar_e = torch.zeros((Ny - 1, 1), dtype=float)
        for iy in range(Ny - 1):
            if Z[Nx - 1, iy] > 0.0 & Z[Nx - 1, iy + 1] > 0.0:
                cff = Z[Nx - 1, iy] + H[0, Nx - 1, iy] + Z[Nx - 1, iy + 1] + H[0, Nx - 1, iy + 1]
                cff1 = N[0, Nx - 1, iy] / D_N[Nx - 1, iy] - N[0, Nx - 2, iy] / D_N[Nx - 2, iy]
                cff2 = Z[Nx - 1, iy] + H[1, Nx - 1, iy] + Z[Nx - 1, iy + 1] + H[1, Nx - 1, iy + 1]
                vbar_e[iy] = N[0, Nx - 1, iy] / D_N[Nx - 1, iy] * cff - 2 * params.CC1[Nx - 1, iy] * np.sqrt(params.g * cff * 0.5) * cff1
                vbar_e[iy] = vbar_e[iy] / cff2
                N[1, Nx - 1, iy] = vbar_e[iy] * D_N[Nx - 1, iy]
    
    # Southern side
    if obc_v2d[1] == 'Fla':
        vbar_s = torch.zeros((Nx, 1), dtype=float)
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
        vbar_s = torch.zeros((Nx, 1), dtype=float)
        for ix in range(Nx):
            if Z[ix, 0] > params.dry_limit:
                vbar_s[ix] = -1.0 * np.sqrt(params.g / (Z[ix, 0] + H[1, ix, 0] + eps)) * H[1, ix, 0]
                N[1, ix, 0] = vbar_s[ix] * D_N[ix, 0]
    
    # northern side
    if obc_v2d[3] == 'Fla':
        vbar_n = torch.zeros((Nx, 1), dtype=float)
        for ix in range(Nx):
            if flux_sign[ix, Ny - 3] != 999:
                bry_pgr = -params.g * (z_n[ix] - H[0, ix, Ny -3]) * 0.5 * params.CC2
                bry_cor = 0.0
                bry_pgr = -params.g * (z_n[ix] - H[0, ix, Ny -3]) * 0.5 * params.CC2
                bry_cor = 0.0  # should add
                cff1 = 1.0 / (0.5 * (Z[ix, Ny - 3] + H[0, ix, Ny - 3] + Z[ix, Ny - 2] + H[0, ix, Ny - 2] + eps))
                bry_str = 0.0  # should add surface forcing
                Ce = 1.0 / np.sqrt(params.g * 0.5 * (Z[ix, Ny - 2] + H[0, ix, Ny - 2] + Z[ix, Ny - 3] + H[0, ix, Ny - 3] + eps))
                cff2 = Ce / params.dy
                bry_val = N[0, ix, Ny - 3] / D_N[ix, Ny - 3] + cff2 * (bry_pgr + bry_cor + bry_str)
                Ce = np.sqrt(params.g * cff1)
                vbar_n[ix] = bry_val + Ce * (0.5 * (H[0, ix, Ny - 3] + H[0, ix, Ny - 2]) - z_n[ix])
                N[1, ix, Ny - 2] = vbar_n[ix] * D_N[ix, Ny-2]
    elif obc_v2d[3] == 'Gra':
        N[1, :, Ny - 2] = N[1, :, Ny - 3]
    elif obc_v2d[3] == 'Clo':
        N[1, :, Ny - 2] = 0.0
    elif obc_v2d[3] == 'Rad':
        vbar_n = torch.zeros((Nx, 1), dtype=float)
        for ix in range(Nx):
            if Z[ix, Ny - 1] > params.dry_limit:
                vbar_n[ix] = 1.0 * np.sqrt(params.g / (Z[ix, Ny - 1] + H[1, ix, Ny - 1] + eps)) * H[1, ix, Ny - 1]
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

def interp_bc(data, interp_ratio, step0, step1):
    res = interp_ratio * (data[:, step1] - data[:, step0]) + data[:, step0]
    return res 

def get_current_bc(params, tide_e, tide_w, tide_n, tide_s, time_unit_sum):
    # get tide_z tide_u tide_v in current model time
    # bc_time_in_secs = tide_e["tide_e"]
    tideStep0 = int(time_unit_sum / params.bc_timestep)
    interp_ratio = time_unit_sum / params.bc_timestep - tideStep0   
        
    tide_z = {'east' : interp_bc(tide_e['tide_z'], interp_ratio, tideStep0, tideStep0+1),
              'west' : interp_bc(tide_w['tide_z'], interp_ratio, tideStep0, tideStep0+1),
              'north': interp_bc(tide_n['tide_z'], interp_ratio, tideStep0, tideStep0+1),
              'south': interp_bc(tide_s['tide_z'], interp_ratio, tideStep0, tideStep0+1)
              }              
    tide_u = {'east': interp_bc(tide_e['tide_u'], interp_ratio, tideStep0, tideStep0+1),
              'west': interp_bc(tide_w['tide_u'], interp_ratio, tideStep0, tideStep0+1),
              'north': interp_bc(tide_n['tide_u'], interp_ratio, tideStep0, tideStep0+1),
              'south': interp_bc(tide_s['tide_u'], interp_ratio, tideStep0, tideStep0+1)
              }
    tide_v = {'east': interp_bc(tide_e['tide_v'], interp_ratio, tideStep0, tideStep0+1),
              'west': interp_bc(tide_w['tide_v'], interp_ratio, tideStep0, tideStep0+1),
              'north': interp_bc(tide_n['tide_v'], interp_ratio, tideStep0, tideStep0+1),
              'south': interp_bc(tide_s['tide_v'], interp_ratio, tideStep0, tideStep0+1)
              }
    return tide_z, tide_u, tide_v
#%%    
#if __name__ == '__main__':
params = Params()
device = torch.device('cpu')

dt = params.dt
NT = params.NT
# 得到逐时预报的潮位和潮流
tide_e, tide_w, tide_n, tide_s = get_bd_Pred(params)

# x = torch.linspace(0, params.Lx, params.Nx + 1)
# y = torch.linspace(0, params.Ly, params.Ny + 1)
# X, Y = torch.meshgrid(x, y, indexing='ij')
X = torch.from_numpy(params.grid.xx)
Y = torch.from_numpy(params.grid.yy)

M = torch.zeros((2, params.grid.Nx-1, params.grid.Ny))
N = torch.zeros((2, params.grid.Nx, params.grid.Ny-1))
H = torch.zeros((2, params.grid.Nx, params.grid.Ny))
Z = torch.from_numpy(params.grid.h).to(dtype=H.dtype) 

manning = torch.zeros((params.grid.Nx, params.grid.Ny)) + params.manning
manning[0:3, :] = 10 * manning[0:3, :]
manning[3:, -3:] = 10 * manning[3:, -3:]
manning[-3:, :] = 10 * manning[-3:, :]
manning[:, 0:3] = 10 * manning[:, 0:3]
# H[0] = 1 * torch.exp(-((X-X.max()//2)**2 + (Y-Y.max()//2)**2) / (2 * 50000**2))
#
# Wx,Wy,Pa = generate_wind(X,Y,params)
# Ws = torch.sqrt(Wx**2 + Wy**2)
#
Wx = np.ones((NT,X.shape[0],X.shape[1]))*0 + 0
Wy = np.ones((NT,X.shape[0],X.shape[1]))*0
Wx = torch.from_numpy(Wx)
Wy = torch.from_numpy(Wy)
Pa = Wx * 0 + 1000
#%%
eta_list = list()
u_list = list()
v_list = list()
anim_interval = 1

time_unit_sum = 0.0
tic()
for itime in range(1):        
    print(itime)
    time_unit = 0.0
    meanDtUnit = []
    while time_unit < params.cal_unit:
        print(f"dealing dt={dt}")
        dt = cal_dynamic_timeStep(params, H[0], M[0], N[0])
        if time_unit + dt + params.eps_time > params.cal_unit:
            dt = params.cal_unit - time_unit
            time_unit = params.cal_unit + params.eps_time
        else:
            time_unit += dt
        meanDtUnit.append(dt)
        time_unit_sum += dd(dt) # for bc interp
        params.dt = dt
        params.CC1 = params.dt * torch.from_numpy(params.grid.pm) #self.dt / self.dx
        params.CC2 = params.dt * torch.from_numpy(params.grid.pn) #self.dt / self.dy
        params.CC3 = params.CC1 * params.g
        params.CC4 = params.CC2 * params.g
        #params.itime=itime
        
        #print(f"stepping!")
        tide_z, tide_u, tide_v = get_current_bc(params, tide_e, tide_w, tide_n, tide_s, time_unit_sum)
        H_update = mass_cartesian_torch(H, Z, M, N, params, tide_z)
        M_update, N_update = momentum_nonlinear_cartesian_torch(H_update, Z, M, N, Wx[itime], Wy[itime], Pa[itime],params, tide_z, tide_u, tide_v, manning)            
        H[0] = H_update[1]
        M[0] = M_update[1]
        N[0] = N_update[1]
        eta_list.append(dd(H_update[1]))
        u_list.append(dd(M_update[1]))
        v_list.append(dd(N_update[1]))            
    timeNow = params.t_date_0 + datetime.timedelta(seconds=time_unit_sum)
    print(f"nt = {itime} / {params.NT}, {timeNow.strftime('%Y-%m-%d %H:%M:%S')}, dt = {np.mean(np.asarray(meanDtUnit)):.2f} s")
    
    if itime%anim_interval==0:
        #mag = np.sqrt( rho2v(M_update[1])**2+rho2u(N_update[1])**2)
        pcolor(params.grid.lon, params.grid.lat, eta_list[-1], vmin=-3,vmax=3, cmap=plt.cm.RdBu_r)
        
        # plot(params.grid.lon[113, 155], params.grid.lat[113, 155], 'ro')
        colorbar()
        #quiver(X[:-1,:-1],Y[:-1,:-1],rho2v(M_update[1]),rho2u(N_update[1]), scale=100)
        #contour(X[:-1,:-1],Y[:-1,:-1],mag)
        xlabel("x [m]", fontname="serif", fontsize=12)
        ylabel("y [m]", fontname="serif", fontsize=12)
        title("Stage at an instant in time: " + f"{timeNow.strftime('%Y-%m-%d %H:%M:%S')}")
        axis('equal')
        #fig.savefig("ele_Rad" + f"{itime}" + "s.jpg", format='jpg', dpi=300, bbox_inches='tight')
        show()

toc()        
    
    
    # ds_out = xr.Dataset(
    #         {'eta': (['time','lat', 'lon'], np.array(eta_list))},
    #         coords={
    #             'lat': x,
    #             'lon': y,
    #             'time': np.arange(len(eta_list))}  )
    # ds_out.to_netcdf('out.circulation.nc')
    #
    # eta = np.array(eta_list)
    # ds = xr.open_dataset('rst.channel.Move3050.nc')
    # eta_true = ds['zeta'][:]
    # eta_true = np.swapaxes(eta_true,1,2)
    # time2plot = [1, itime//10, itime//5, itime//3, itime//2, itime-2]
    # for itime in time2plot:
    #     ind2plot = itime
    #     fig, ax = plt.subplots(1, 1)
    #     # plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 17)
    #     xlabel("x [m]", fontname="serif", fontsize=12)
    #     ylabel("y [m]", fontname="serif", fontsize=12)
    #     title("Stage at an instant in time: " + f"{itime*dt}" + " second")
    # # pmesh = plt.pcolormesh(X, Y, eta_list[500], vmin=-0.7 * np.abs(eta_list[int(len(eta_list) / 2)]).max(),
    # #                        vmax=np.abs(eta_list[int(len(eta_list) / 2)]).max(), cmap=plt.cm.RdBu_r)
    #     ubar = u_list[ind2plot]
    #     vbar = v_list[ind2plot]
    #     pcolor(X, Y, eta_list[ind2plot], vmin=-.2,vmax=.2, cmap=plt.cm.RdBu_r)
    #     colorbar()
    #     quiver(X[::3,::3],Y[::3,::3],ubar[::3,::3],vbar[::3,::3], scale=70)
    #     #quiver(X[::3,::3],Y[::3,::3],(Wx/Ws)[ind2plot,::3,::3],(Wy/Ws)[ind2plot,::3,::3], scale=70)
    #     axis('equal')
    #     savefig("ele_Rad" + f"{itime*dt}" + "s.jpg", format='jpg', dpi=300, bbox_inches='tight')
    #     #show()
    #     figure()
    #     plot(eta[ind2plot,:,25])
    #     plot(eta_true[ind2plot,:,25],'r--')
    #     legend(('Model','ROMS'))
    #     title("Profile at Y=200km at time: " + f"{itime*dt}" + " second")
    #     savefig("ele_profile " + f"{itime*dt}" + "s.jpg", format='jpg', dpi=300, bbox_inches='tight')
        
        
        
    # x2plot = [2, 10, 30, 50, 70, 90, 95, 120, 150, 180, 220]
    # x2plot = [20, 30]
    # for ix in x2plot:
    #     plot_stage_Point(eta_list, ix)
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