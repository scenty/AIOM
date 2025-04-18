
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

