
def momentum_nonlinear_cartesian_test(H, Z, M, N, Wx, Wy, Pa, params, manning):
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
    m0 = M[0].clone().detach() #LWF
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
    
    # Nu is applied here and in the friction below
    N_exp = torch.cat( (N[0,:,0:1], N[0], N[0,:,-1:]), dim = 1)
    Nu = rho2u(v2rho(N_exp))   #at u-point
    
    # 底摩擦
    mask_fs999_M = (flux_sign_M == 999)
    #M_val = torch.where(mask_fs999_M, torch.zeros_like(M_val), M_val)
    #by LWF
    friction_mask = (D0 > FrictionDepthLimit)
    epsilon = 1e-9
    Cf_u = Cf[:-1,:]                
    #Nu was generated before
    Fx = g * Cf_u**2 / (D0**2.33 + 1e-9) * torch.sqrt(m0**2 + Nu**2) * m0
    #by LWF, when optimize, do not use clamping, replacing, or non-torch operations
    # Fx = g * Cf_u**2 / (D0**2.33 + 1e-9) * torch.sqrt(torch.clamp(m0**2 + Nu**2, min=epsilon)) * m0
    # Fx = torch.where(torch.abs(dt*Fx)>torch.abs(m0), m0/dt, Fx)
    # Fx = torch.where(friction_mask, Fx, torch.zeros_like(Fx))
        
    phi_M = 1.0 - CC1 * torch.min(torch.abs(m0 / torch.clamp(D0, min=MinWaterDepth)), torch.sqrt(g*D0))
    M_new = M.clone().detach() #by LWF    
    M_new[1] = (phi_M*m0 + 0.5*(1.0 - phi_M)*(m1 + m2)        #implicit friction
            + dt  * ( sustr + f_cor * Nu)                  #wind stress and coriolis
            - CC1 * ududx                                  #u advection
            - CC2 * vdudy                                  #v advection
            - CC3 * D0*(h2u_M - h1u_M) + Pre_grad_x        #pgf
            - dt  * Fx)                                    #friction
    
    #pcolor(rho2u(X),rho2u(Y),ududx);colorbar();show()
    #pcolor(rho2u(X),rho2u(Y),vdudy);colorbar();show()   
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
    
    # Mv is applied here and in the friction below
    M_exp = torch.cat( (M[0,0:1],M[0], M[0,-1:]), dim =0)
    Mv = rho2u(v2rho(M_exp)) #at v-point, [ 1 ~ Nx-1, 1 ~ Ny-1]
    #Mv = F.pad( rho2u(rho2v( M[0])) , (0,0,1,1)) #pad for up and down
    #底摩擦
    mask_fs999_N = (flux_sign_N == 999)
    #N_val = torch.where(mask_fs999_N, torch.zeros_like(N_val), N_val)
    friction_maskN = (D0N > FrictionDepthLimit)
    epsilon = 1e-9
    Cf_v = Cf[:, :-1]
    Fy = g * Cf_v**2 / (D0N**2.33 + 1e-9) * torch.sqrt(Mv**2 + n0**2) * n0
    #by LWF, when optimize, do not use clamping, replacing, or non-torch operations
    #Fy = g * Cf_v**2 / (D0N**2.33 + 1e-9) * torch.sqrt(torch.clamp(Mv**2 + n0**2, min=epsilon)) * n0
    #Fy = torch.where(torch.abs(dt*Fy)>torch.abs(n0), n0/dt, Fy)
    #Fy = torch.where(friction_maskN, Fy, torch.zeros_like(Fy))
    # 
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
    #M_new = bcond_u2D_torch(H, Z, M_new, D_M, z_w, z_e, params)
    #N_new = bcond_v2D_torch(H, Z, N_new, D_N, z_s, z_n, params)
    assert not torch.any(torch.isnan(M_new))
    assert not torch.any(torch.isnan(N_new))
    
    return M_new, N_new, Fx, Fy