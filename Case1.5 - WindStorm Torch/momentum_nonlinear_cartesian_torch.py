def momentum_nonlinear_cartesian_torch(H, Z, M, N, params):
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
    manning = params.manning
    dt = params.dt
    phi = params.centerWeighting0
    Nx = params.Nx
    Ny = params.Ny
    
    # 重构流深
    D_M, D_N = reconstruct_flow_depth_torch(H, Z, M, N, params)
    # get forcing
    windSpeed = np.sqrt(Wx * Wx + Wy * Wy)
    sustr = rho_air / rho_water * Cd * windSpeed * Wx
    svstr = rho_air / rho_water * Cd * windSpeed * Wy
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

     
    mask_fs999_M = (flux_sign_M == 999)
    M_val = torch.where(mask_fs999_M, torch.zeros_like(M_val), M_val)

    # 底摩擦
    friction_mask = (D0 > FrictionDepthLimit)
    Cf = manning
    MM = m0**2
    NN = torch.zeros_like(m0)
    Fx = g * Cf**2 / (D0**2.33 + 1e-9) * torch.sqrt(MM + NN) * m0
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
    Fy = g * Cf**2 / (D0N**2.33 + 1e-9) * torch.sqrt(MM_N + NN_N) * n0
    Fy = torch.where(torch.abs(dt*Fy)>torch.abs(n0), n0/dt, Fy)
    N_val = N_val - torch.where(friction_maskN, dt*Fy, torch.zeros_like(Fy))

    N_new = N.clone()
    N_new[1] = N_val

    return M_new, N_new,Fx,Fy
    
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

