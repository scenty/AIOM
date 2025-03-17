def momentum_nonlinear_cartesian(H, Z, M, N, Wx, Wy, Pa, params):
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

    D_M, D_N = reconstruct_flow_depth(H, Z, M, N, params)
    # get sustr svstr
    windSpeed = np.sqrt(Wx * Wx + Wy * Wy)
    sustr = rho_air / rho_water * Cd * windSpeed * Wx
    svstr = rho_air / rho_water * Cd * windSpeed * Wy

    # Rest of the function remains the same, using the extracted parameters
    # ================= 处理M分量 (i方向通量) =================
    nstartx = 1
    nendx = Nx-1
    nstarty = 1
    nendy = Ny
    for ix in range(nstartx, nendx):
        for iy in range(nstarty, nendy):
            z1 = Z[ix, iy]
            z2 = Z[ix + 1, iy]
            h1 = H[1, ix, iy]
            h2 = H[1, ix + 1, iy]
            P1 = Pa[ix, iy]
            P2 = Pa[ix + 1, iy]
            
            flux_sign, h1_update, h2_update = check_flux_direction(z1, z2, h1, h2, params)
            if flux_sign == 999:
                M[1, ix, iy] = 0.0
            else:
                # !///solve NSWE///!
                MU1 = 0.0
                MU2 = 0.0
                NU1 = 0.0
                NU2 = 0.0
                m0 = M[0, ix, iy]
                D0 = D_M[ix, iy]
                m1, D1 = M[0, ix - 1, iy], D_M[ix - 1, iy]
                m2, D2 = M[0, ix + 1, iy], D_M[ix + 1, iy]
                m3, D3 = M[0, ix, iy - 1], D_M[ix, iy - 1]
                n3, n4 = N[0, ix, iy - 1], N[0, ix + 1, iy - 1]
                m5, D5 = M[0, ix, iy + 1], D_M[ix, iy + 1]
                n0, n2 = N[0, ix, iy], N[0, ix+ 1, iy]
                # !/// extra terms n5, n6, n7, n8 ///!
                if iy == 1:
                    n7, n8 = N[0, ix, iy - 1], N[0, ix + 1, iy - 1]  # attention !!!
                else:
                    n7, n8 = N[0, ix, iy - 2], N[0, ix + 1, iy - 2] # attention !!!
                if iy == Ny - 1:
                    n5, n6 = N[0, ix, iy], N[0, ix + 1, iy]
                else:
                    n5, n6 = N[0, ix, iy + 1], N[0, ix + 1, iy + 1]
                # Upwind scheme
                if m0 >= 0.0:
                    if D1 > Dlimit:
                        MU1 = m1 ** 2 / D1
                    if D0 > Dlimit:
                        MU2 = m0 ** 2 / D0
                else:
                    if D0 > Dlimit:
                        MU1 = m0 ** 2 / D0
                    if D2 > Dlimit:
                        MU2 = m2 ** 2 / D2

                if (n0 + n2 + n3 + n4) >= 0.0:
                    if D3 > Dlimit:
                        NU1 = 0.25 * m3 * (n3 + n4 + n7 + n8) / D3
                    if D0 > Dlimit:
                        NU2 = 0.25 * m0 * (n0 + n2 + n3 + n4) / D0
                else:
                    if D0 > Dlimit:
                        NU1 = 0.25 * m0 * (n0 + n2 + n3 + n4) / D0
                    if D5 > Dlimit:
                        NU2 = 0.25 * m5 * (n0 + n2 + n5 + n6) / D5
                Pre_grad_x = CC1 * D0 * (P2 - P1) / rho_water
                # Flux-centered
                phi = 1.0 - CC1 * min(np.abs(m0 / max(D0, MinWaterDepth)), np.sqrt(g * D0))
                M[1, ix, iy] = (phi * m0 + 0.5 * (1.0 - phi) * (m1 + m2)
                            - (CC3 * D0 * (h2_update - h1_update) + Pre_grad_x)
                            + dt * ( 0.5 * (sustr[ix, iy] + sustr[ix + 1, iy]) + f_cor * (n0 + n2 + n3 + n4))
                            - CC1 * (MU2 - MU1)
                            - CC2 * (NU2 - NU1) )
                # Bottom friction
                if D0 > FrictionDepthLimit:
                    MM = m0 ** 2
                    NN = (0.25 * (n0 + n2 + n3 + n4)) ** 2
                    Cf = manning
                    Fx = g * Cf ** 2 / (D0 ** 2.33 + 1e-9) * np.sqrt(MM + NN) * m0
                    if abs(dt * Fx) > abs(m0):
                        Fx = m0 / dt
                    M[1, ix, iy] -= dt * Fx
                # Check direction of M
                if flux_sign != 0 and M[0, ix, iy] * flux_sign < 0.0:
                    M[1, ix, iy] = 0.0
    # ================= 处理N分量 (i方向通量) =================
    nstartx = 1
    nendx = params.Nx
    nstarty = 1
    nendy = params.Ny - 1
    for ix in range(nstartx, nendx):
        for iy in range(nstarty, nendy):
            z1 = Z[ix, iy]
            z2 = Z[ix, iy + 1]
            h1 = H[1, ix, iy]
            h2 = H[1, ix, iy + 1]
            P1 = Pa[ix, iy]
            P2 = Pa[ix, iy + 1]
            flux_sign, h1_update, h2_update = check_flux_direction(z1, z2, h1, h2, params)
            if flux_sign == 999:
                N[1, ix, iy] = 0.0
            else:
                # !///solve NSWE///!
                NV1 = 0.0
                NV2 = 0.0
                MV1 = 0.0
                MV2 = 0.0
                n0 = N[0, ix, iy]
                D0 = D_N[ix, iy]
                n3, D3 = N[0, ix, iy - 1], D_N[ix, iy - 1]
                n5, D5 = N[0, ix, iy + 1], D_N[ix, iy + 1]
                n1, D1 = N[0, ix - 1, iy], D_N[ix - 1, iy]
                m1, m4 = M[0, ix - 1, iy], M[0, ix - 1, iy + 1]

                n2, D2 = N[0, ix + 1, iy], D_N[ix + 1, iy]
                m0, m5 = M[0, ix, iy], M[0, ix, iy + 1]
                #!/// extra terms m2, m6, m7, m8///!
                if ix == 1:
                    m7, m8 = M[0, ix - 1, iy], M[0, ix - 1, iy + 1] # attention here
                else:
                    m7, m8 = M[0, ix - 2, iy], M[0, ix - 2, iy + 1]
                if ix == Nx - 1:
                    m2, m6 = M[0, ix, iy], M[0, ix, iy + 1]
                else:
                    m2, m6 = M[0, ix + 1, iy], M[0, ix + 1, iy + 1]
                if n0 >= 0.0:
                    if D3 > Dlimit:
                        NV1 = n3 ** 2 / D3
                    if D0 > Dlimit:
                        NV2 = n0 ** 2 / D0
                else:
                    if D0 > Dlimit:
                        NV1 = n0 ** 2 / D0
                    if D5 > Dlimit:
                        NV2 = n5 ** 2 / D5

                if m0 + m1 + m4 + m5 >= 0.0:
                    if D1 > Dlimit:
                        MV1 = 0.25 * n1 * (m1 + m4 + m7 + m8) / D1
                    if D0 > Dlimit:
                        MV2 = 0.25 * n0 * (m0 + m1 + m4 + m5) / D0
                else:
                    if D0 > Dlimit:
                        MV1 = 0.25 * n0 * (m0 + m1 + m4 + m5) / D0
                    if D2 > Dlimit:
                        MV2 = 0.25 * n2 * (m0 + m2 + m5 + m6) / D2
                Pre_grad_y = CC2 * D0 * (P2 - P1) / rho_water

                phi = 1.0 - CC2 * min(np.abs(n0 / max(D0, MinWaterDepth)),
                                      np.sqrt(g * D0))

                N[1, ix, iy] = (phi * n0 + 0.5 * (1.0 - phi) * (n3 + n5)
                                                  - (CC4 * D0 * (h2_update - h1_update) + Pre_grad_y)
                                                  + dt * (0.5 * (svstr[ix, iy] + svstr[ix, iy + 1]) - f_cor * (m0 + m1 + m4 + m5))
                                                  - CC2 * (NV2 - NV1)
                                                  - CC1 * (MV2 - MV1))
                if D0 > FrictionDepthLimit:
                    MM, NN = (0.25 * (m0 + m1 + m4 + m5)) ** 2, n0 ** 2
                    Cf = manning
                    Fy = g * Cf ** 2 / (D0 ** 2.33 + 1e-9) * np.sqrt(MM + NN) * n0
                    if np.abs(dt * Fy) > abs(n0):
                        Fy = n0 / dt
                    N[1, ix, iy] -= dt * Fy
                if flux_sign != 0 and N[1, ix, iy] * flux_sign < 0.0:
                    N[1, ix, iy] = 0.0
        # apply boundary condition
    z_w = 0.05 * np.sin(2 * np.pi / 0.25 * itime * dt) * (np.zeros((Ny + 1, 1)) + 1)
    z_s = 0
    z_n = 0
    z_e = 0
    M_update = bcond_u2D(H_update, Z, M, D_M, z_w, z_e, params)
    N_update = bcond_v2D(H_update, Z, N, D_N, z_s, z_n, params)
    return M_update, N_update

def reconstruct_flow_depth(H, Z, M, N, params):
    """
    向量化重构流深计算函数
    处理动量分量M和N的流深D_M/D_N计算
    """
    D_M = np.zeros_like(M[0, :])
    D_N = np.zeros_like(N[0, :])
    # ================= 处理M分量 (i方向通量) =================
    nstartx = 0
    nendx = params.Nx
    nstarty = 0
    nendy = params.Ny + 1
    for ix in range(nstartx, nendx):
        for iy in range(nstarty, nendy):
            z1 = Z[ix, iy]
            z2 = Z[ix + 1, iy]
            h1 = H[1, ix, iy]
            h2 = H[1, ix + 1, iy]
            m0 = M[0, ix, iy]
            flux_sign, h1_update, h2_update = check_flux_direction(z1, z2, h1, h2,params)
            if flux_sign == 999:
                D_M[ix, iy] = 0.0
            elif flux_sign == 0:
                if m0 >= 0.0:
                    D_M[ix, iy] = 0.5 * (z1 + z2) + h1_update
                else:
                    D_M[ix, iy] = 0.5 * (z1 + z2) + h2_update
            else:
                D_M[ix, iy] = 0.5 * (z1 + z2) + np.max( (h1_update, h2_update) )
    # ================= 处理N分量 (i方向通量) =================
    nstartx = 0
    nendx = params.Nx + 1
    nstarty = 0
    nendy = params.Ny
    for ix in range(nstartx, nendx):
        for iy in range(nstarty, nendy):
            z1 = Z[ix, iy]
            z2 = Z[ix, iy + 1]
            h1 = H[1, ix, iy]
            h2 = H[1, ix, iy + 1]
            n0 = N[0, ix, iy]
            flux_sign, h1_update, h2_update = check_flux_direction(z1, z2, h1, h2,params)
            if flux_sign == 999:
                D_N[ix, iy] = 0.0
            elif flux_sign == 0:
                if n0 >= 0.0:
                    D_N[ix, iy] = 0.5 * (z1 + z2) + h1_update
                else:
                    D_N[ix, iy] = 0.5 * (z1 + z2) + h2_update
            else:
                D_N[ix, iy] = 0.5 * (z1 + z2) + np.max( (h1_update, h2_update) )

    return D_M, D_N

def check_flux_direction(z1, z2, h1, h2, params):
    dry_limit = params.dry_limit
    if z1 <= -dry_limit or z2 <= -dry_limit:
        return 999,h1,h2
    if (z1 + h1) <=0.0 or (z2 + h2) <= 0.0:
        if (z1 + h1) <= 0.0 and (z2 + h2) <= 0.0:
            return 999,h1,h2
        elif (z1 > z2) and (z1 + h1 > 0.0) and (z2 + h1 <=0.0):
            return 999,h1,h2
        elif (z2 > z1) and (z2 + h2 > 0.0) and (z1 + h2 <=0.0):
            return 999,h1,h2
    if (z1 <= z2) and (z1 + h1) > 0.0 and (z1 + h2)<= 0.0:
        flux_sign = 1
        h2 = -z1
    elif (z1 >= z2) and (z2 + h2) <= 0.0 and (z2 + h1) > 0.0:
        flux_sign = 2
        h2 = -z2
    elif (z2 <= z1) and (z2 + h2) > 0.0 and (z2 + h1) <= 0.0:
        flux_sign = -1
        h1 = -z2
    elif (z2 >= z1) and (z1 + h1) <= 0.0 and (z1 + h2) > 0.0:
        flux_sign = -2
        h1 = -z1
    else:
        flux_sign = 0

    return flux_sign, h1, h2
