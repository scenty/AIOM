
def bcond_zeta_vec(H, Z, params):
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
        #for iy in range(Ny): #removing Ny loop
        Cx = CC1 * torch.sqrt(g * (Z[1] + H0[1]))
        H1 = (1.0 - Cx) * H0[0] + Cx * H0[1]
    elif obc_ele[0] == 'Cha_i':  # implicit Chapman boundary condition.
        for iy in range(Ny):
            Cx = CC1 * torch.sqrt(g * (Z[1, iy] + H0[1, iy]))
            cff2 = 1.0 / (1.0 + Cx)
            H1[0, iy] = cff2 * (H0[0, iy] + Cx * H1[1, iy])
    elif obc_ele[0] == 'Gra':
        H1[0] = H1[1] # by LWF
    elif obc_ele[0] == 'Clo':
        H1[0, :] = 0.0
    elif obc_ele[0] == 'Rad':
        #for iy in range(Ny): removing the Ny loop
        H1[0, :] = H0[0, :] - 2 * CC1 / torch.sqrt(g * (Z[0, :] + H0[0, :])) * (H0[0, :] - H0[1, :])
    
    # --- Eastern Boundary (iy loop -> vectorized) ---
    if obc_ele[2] == 'Cha_e':
        Cx = CC1 * torch.sqrt(g * (Z[Nx-2, :] + H0[Nx-2, :]))
        H1[Nx-1, :] = (1.0 - Cx) * H0[Nx-1, :] + Cx * H0[Nx-2, :]
    elif obc_ele[2] == 'Cha_i':
        Cx = CC1 * torch.sqrt(g * (Z[Nx-2, :] + H0[Nx-2, :]))
        cff2 = 1.0 / (1.0 + Cx)
        H1[Nx-1, :] = cff2 * (H0[Nx-1, :] + Cx * H1[Nx-2, :])
    elif obc_ele[2] == 'Gra':
        H1[Nx-1, :] = H1[Nx-2, :]
    elif obc_ele[2] == 'Clo':
        H1[Nx-1, :] = 0.0
    elif obc_ele[2] == 'Rad':
        Cx = 2 * CC1 / torch.sqrt(g * (Z[Nx-1, :] + H0[Nx-1, :]))
        H1[Nx-1, :] = H0[Nx-1, :] - Cx * (H0[Nx-1, :] - H0[Nx-2, :])
        
    # --- Southern Boundary (ix loop -> vectorized) ---
    if obc_ele[1] == 'Cha_e':
        Ce = CC2 * torch.sqrt(g * (Z[:, 1] + H0[:, 1]))
        H1[:, 0] = (1.0 - Ce) * H0[:, 0] + Ce * H0[:, 1]
    elif obc_ele[1] == 'Cha_i':
        Ce = CC2 * torch.sqrt(g * (Z[:, 1] + H0[:, 1]))
        cff2 = 1.0 / (1.0 + Ce)
        H1[:, 0] = cff2 * (H0[:, 0] + Ce * H1[:, 1])
    elif obc_ele[1] == 'Gra':
        H1[:, 0] = H1[:, 1]
    elif obc_ele[1] == 'Clo':
        H1[:, 0] = 0.0
    elif obc_ele[1] == 'Rad':
        Ce = 2.0 * CC2 * torch.sqrt(g * (Z[:, 0] + H0[:, 0]))
        H1[:, 0] = H0[:, 0] - Ce * (H0[:, 0] - H0[:, 1])
    
    # --- Northern Boundary (ix loop -> vectorized) ---
    if obc_ele[3] == 'Cha_e':
        Ce = CC2 * torch.sqrt(g * (Z[:, Ny-2] + H0[:, Ny-2]))
        H1[:, Ny-1] = (1.0 - Ce) * H0[:, Ny-1] + Ce * H0[:, Ny-2]
    elif obc_ele[3] == 'Cha_i':
        Ce = CC2 * torch.sqrt(g * (Z[:, Ny-2] + H0[:, Ny-2]))
        cff2 = 1.0 / (1.0 + Ce)
        H1[:, Ny-1] = cff2 * (H0[:, Ny-1] + Ce * H1[:, Ny-2])
    elif obc_ele[3] == 'Gra':
        H1[:, Ny-1] = H1[:, Ny-2]
    elif obc_ele[3] == 'Clo':
        H1[:, Ny-1] = 0.0
    elif obc_ele[3] == 'Rad':
        Ce = 2.0 * CC2 * torch.sqrt(g * (Z[:, Ny-1] + H0[:, Ny-1]))
        H1[:, Ny-1] = H0[:, Ny-1] - Ce * (H0[:, Ny-1] - H0[:, Ny-2])

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
    obc_u2d = params.obc_u2d
    if H.ndim == 3:
        Nx, Ny = H.shape[1], H.shape[2] # 此处： L = Nx, M = Ny, Lm = Nx - 1, Mm = Ny - 1
        M_new = M.clone()
    else: # H.ndim==2
        Nx, Ny = H.shape[0], H.shape[1]
        M_new = M.clone()
    
    # ---------------------------
    # 西侧边界更新：更新 M_new[1, 0, :]（更新第0行）
    if obc_u2d[0] == 'Fla':
        ubar_w_list = []
        for iy in range(Ny):
            bry_pgr = -params.g * (H[0, 1, iy] - z_w[iy]) * 0.5 * params.CC1
            bry_cor = 0.0
            cff1 = 1.0 / (0.5 * (Z[0, iy] + H[0, 0, iy] + Z[1, iy] + H[0, 1, iy]))
            bry_str = 0.0
            Cx = 1.0 / torch.sqrt(params.g * 0.5 * (Z[0, iy] + H[0, 0, iy] + Z[1, iy] + H[0, 1, iy]))
            cff2 = Cx / params.dx
            bry_val = M[0, 1, iy] / D_M[1, iy] + cff2 * (bry_pgr + bry_cor + bry_str)
            Cx = torch.sqrt(params.g * cff1)
            new_val = bry_val - Cx * (0.5 * (H[0, 0, iy] + H[0, 1, iy]) - z_w[iy])
            ubar_w_list.append(new_val)
        ubar_w_tensor = torch.stack(ubar_w_list, dim=0)
        old_row = M_new[1][0, :]
        new_row = ubar_w_tensor * D_M[0, :]
        M_new_row = torch.cat([new_row.unsqueeze(0), M_new[1][1:, :].clone()], dim=0)
        M_new = torch.stack([M_new[0], M_new_row], dim=0)
    
    elif obc_u2d[0] == 'Gra':
        M_new[1] = F.pad( M_new[1:2, 1:], (0,0,1,0), mode='replicate') #pad for West, up 
    
    elif obc_u2d[0] == 'Clo':
        M_new[1] = F.pad( M_new[1, 1:], (0,0,1,0) ) #pad for West, up 
    
    elif obc_u2d[0] == 'Rad':
        ubar_w_list = []
        for iy in range(Ny):
            new_val = -1.0 * torch.sqrt(params.g / (Z[0, iy] + H[1, 0, iy])) * H[1, 0, iy]
            ubar_w_list.append(new_val)
        ubar_w_tensor = torch.stack(ubar_w_list, dim=0)
        old_row = M_new[1][0, :]
        new_row = ubar_w_tensor * D_M[0, :]
        M_new_row = torch.cat([new_row.unsqueeze(0), M_new[1][1:, :].clone()], dim=0)
        M_new = torch.stack([M_new[0], M_new_row], dim=0)
    
    # ---------------------------
    # 东侧边界更新：更新 M_new[1, Nx-2, :]（更新第 Nx-2 行）
    if obc_u2d[2] == 'Fla':
        ubar_e_list = []
        for iy in range(Ny):
            bry_pgr = params.g * (z_e[iy] - H[0, Nx-3, iy]) * 0.5 * params.CC1
            bry_cor = 0.0
            cff1 = 1.0 / (0.5 * (Z[Nx-3, iy] + H[0, Nx-3, iy] + Z[Nx-2, iy] + H[0, Nx-2, iy]))
            bry_str = 0.0
            Cx = 1.0 / torch.sqrt(params.g * 0.5 * (Z[Nx-2, iy] + H[0, Nx-2, iy] + Z[Nx-3, iy] + H[0, Nx-3, iy]))
            cff2 = Cx / params.dx
            bry_val = M[0, Nx-3, iy] / D_M[Nx-3, iy] + cff2 * (bry_pgr + bry_cor + bry_str)
            Cx = torch.sqrt(params.g * cff1)
            new_val = bry_val + Cx * (0.5 * (H[0, Nx-3, iy] + H[0, Nx-2, iy]) - z_e[iy])
            ubar_e_list.append(new_val)
        ubar_e_tensor = torch.stack(ubar_e_list, dim=0)
        old_row = M_new[1][Nx-2, :]
        new_row = ubar_e_tensor * D_M[Nx-2, :]
        M_row_above = M_new[1][:Nx-2, :].clone()
        M_row_below = M_new[1][Nx-1:, :].clone()
        M_new_row = torch.cat([M_row_above, new_row.unsqueeze(0), M_row_below], dim=0)
        M_new = torch.stack([M_new[0], M_new_row], dim=0)
    
    elif obc_u2d[2] == 'Gra':        
        M_new[1] = F.pad( M_new[1:2,:-1], (0,0,0,1), mode='replicate') #pad for East
        
    elif obc_u2d[2] == 'Clo':
        M_new[1] = F.pad( M_new[1,:-1], (0,0,0,1) ) #pad for East
    
    elif obc_u2d[2] == 'Rad':
        ubar_e_list = []
        for iy in range(Ny):
            new_val = 1.0 * torch.sqrt(params.g / (Z[Nx-1, iy] + H[1, Nx-1, iy])) * H[1, Nx-1, iy]
            ubar_e_list.append(new_val)
        ubar_e_tensor = torch.stack(ubar_e_list, dim=0)
        old_row = M_new[1][Nx-2, :]
        new_row = ubar_e_tensor * D_M[Nx-2, :]
        M_row_above = M_new[1][:Nx-2, :].clone()
        M_row_below = M_new[1][Nx-1:, :].clone()
        M_new_row = torch.cat([M_row_above, new_row.unsqueeze(0), M_row_below], dim=0)
        M_new = torch.stack([M_new[0], M_new_row], dim=0)
        
    # ---------------------------
    # 南侧边界更新：更新 M_new[1, :, 0] （对 x 方向，更新前 Nx-1 个位置）
    if obc_u2d[1] == 'Fla':
        ubar_s_list = []
        for ix in range(Nx - 1):
            cff   = params.dt * 0.5 / params.dy
            cff1  = torch.sqrt(params.g * 0.5 * (Z[ix, 1] + H[0, ix, 1] + Z[ix+1, 1] + H[0, ix+1, 1]))
            Ce    = cff * cff1
            cff2  = 1.0 / (1.0 + Ce)
            new_val = cff2 * (M[0, ix, 0] / D_M[ix, 0] + Ce * M[1, ix, 1] / D_M[ix, 1])
            ubar_s_list.append(new_val)
        ubar_s_tensor = torch.stack(ubar_s_list, dim=0)  # shape: (Nx-1,)
        # 取原南侧边界列（M_new[1][:,0]），用新计算的前 Nx-1 个值替换，后面的保持原状
        old_col = M_new[1][:, 0]
        new_col = torch.cat([ubar_s_tensor * D_M[:Nx-1, 0], old_col[Nx-1:].clone()], dim=0)
        # 重组 M_new[1]：将第一列替换为 new_col，其它列保持不变
        M_new_row = torch.cat([new_col.unsqueeze(1), M_new[1][:, 1:]], dim=1)
        M_new = torch.stack([M_new[0], M_new_row], dim=0)
    
    elif obc_u2d[1] == 'Gra':
        M_new[1] = F.pad( M_new[1:2, :, 1:], (1,0,0,0), mode='replicate') #pad for South, fast by 20%
    
    elif obc_u2d[1] == 'Clo':
        # new_col = torch.zeros((Nx-1,), dtype=M_new.dtype, device=M_new.device).unsqueeze(1)
        # M_new_row = torch.cat([new_col, M_new[1, :, 1:]], dim=1)
        # M_new = torch.stack([M_new[0], M_new_row], dim=0)

        M_new[1] = F.pad( M_new[1, :, 1:], (1,0,0,0) ) #pad for South, fast by 20%
    
    elif obc_u2d[1] == 'Rad':
        ubar_s_list = []
        for ix in range(Nx - 1):
            cff   = Z[ix, 0] + H[0, ix, 0] + Z[ix+1, 0] + H[0, ix+1, 0]
            cff1  = M[0, ix, 0] / D_M[ix, 0] - M[0, ix, 1] / D_M[ix, 1]
            cff2  = Z[ix, 0] + H[1, ix, 0] + Z[ix+1, 0] + H[1, ix+1, 0]
            new_val = (M[0, ix, 0] / D_M[ix, 0] * cff - 2 * params.CC2 * torch.sqrt(params.g * cff * 0.5) * cff1) / cff2
            ubar_s_list.append(new_val)
        ubar_s_tensor = torch.stack(ubar_s_list, dim=0)
        old_col = M_new[1][:, 0]
        new_col = torch.cat([ubar_s_tensor * D_M[:Nx-1, 0], old_col[Nx-1:].clone()], dim=0)
        M_new_row = torch.cat([new_col.unsqueeze(1), M_new[1][:, 1:]], dim=1)
        M_new = torch.stack([M_new[0], M_new_row], dim=0)
    
    # ---------------------------
    # 北侧边界更新：更新 M_new[1, :, Ny-1]
    if obc_u2d[3] == 'Fla':
        ubar_n_list = []
        for ix in range(Nx - 1):
            cff   = params.dt * 0.5 / params.dy
            cff1  = torch.sqrt(params.g * 0.5 * (Z[ix, Ny-2] + H[0, ix, Ny-2] + Z[ix+1, Ny-2] + H[0, ix+1, Ny-2]))
            Ce    = cff * cff1
            cff2  = 1.0 / (1.0 + Ce)
            new_val = cff2 * (M[0, ix, Ny-1] / D_M[ix, Ny-1] + Ce * M[1, ix, Ny-2] / D_M[ix, Ny-2])
            ubar_n_list.append(new_val)
        ubar_n_tensor = torch.stack(ubar_n_list, dim=0)
        old_col = M_new[1][:, Ny-1]
        new_col = torch.cat([ubar_n_tensor * D_M[:Nx-1, Ny-1], old_col[Nx-1:].clone()], dim=0)
        left_cols = M_new[1][:, :Ny-1]
        M_new_row = torch.cat([left_cols, new_col.unsqueeze(1)], dim=1)
        M_new = torch.stack([M_new[0], M_new_row], dim=0)
    
    elif obc_u2d[3] == 'Gra':
        M_new[1] = F.pad( M_new[1:2, :, :-1], (0,1,0,0), mode='replicate') #pad for North, fast by 20%

    elif obc_u2d[3] == 'Clo':
        M_new[1] = F.pad( M_new[1, :, :-1], (0,1,0,0) ) #pad for North, fast by 20%
    
    elif obc_u2d[3] == 'Rad':
        ubar_n_list = []
        for ix in range(Nx - 1):
            new_val = 1.0 * torch.sqrt(params.g / (Z[ix, Ny-1] + H[1, ix, Ny-1])) * H[1, ix, Ny-1]
            ubar_n_list.append(new_val)
        ubar_n_tensor = torch.stack(ubar_n_list, dim=0)
        old_col = M_new[1][:, Ny-1]
        new_col = torch.cat([ubar_n_tensor * D_M[:Nx-1, Ny-1], old_col[Nx-1:].clone()], dim=0)
        left_cols = M_new[1][:, :Ny-1]
        M_new_row = torch.cat([left_cols, new_col.unsqueeze(1)], dim=1)
        M_new = torch.stack([M_new[0], M_new_row], dim=0)
        
    return M_new

