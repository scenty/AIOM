import numpy as np
#import vis_tools
import matplotlib.pyplot as plt
#import os
from J65 import *
from matplotlib.pyplot import *
import xarray as xr

def mass_cartesian(H, Z, M, N):
    """
    求解质量方程（优化版）。
    参数：
    Z  : depth
    H  : zeta
    M  : HU
    N  : HV
    """
    M0 = M[0, :]
    N0 = N[0, :]
    mask_dry = Z <= -dry_limit
    H[:, mask_dry] = 0.0
    nstartx = 1
    nendx = Nx
    nstarty = 1
    nendy = Ny
    for ix in range(nstartx, nendx):
        for iy in range(nstarty, nendy):
            if Z[ix, iy] <= - dry_limit:
                H[1, ix, iy] = 0.0
                continue
            m1 = M0[ix-1, iy]
            m2 = M0[ix, iy]
            n1 = N0[ix, iy - 1]
            n2 = N0[ix, iy]
            H[1, ix, iy] = H[0, ix, iy] - CC1 *(m2 - m1) - CC2 * (n2 - n1)

            if (Z[ix, iy] < dry_limit) and (Z[ix, iy] > -dry_limit):
                # Wet -> Dry
                if ((Z[ix, iy] + H[0, ix, iy] > 0.0) and
                        (H[1, ix, iy] - H[0, ix, iy] < 0.0) and
                        (Z[ix, iy] + H[1, ix, iy] <= MinWaterDepth)):

                    if Z[ix, iy] > 0.0:
                        H[1, ix, iy] = -Z[ix, iy]
                    else:
                        H[1, ix, iy] = 0.0
                # Dry -> Wet/Dry
                if (Z[ix, iy] + H[0, ix, iy] <= 0.0):
                    if (H[1, ix, iy] - H[0, ix, iy] > 0.0):  # Dry -> Wet
                        H[1, ix, iy] = H[1, ix, iy] - H[0, ix, iy] - Z[ix, iy]
                    else:  # Dry -> Dry
                        if Z[ix, iy] > 0.0:
                            H[1, ix, iy] = -Z[ix, iy]
                        else:
                            H[1, ix, iy] = 0.0

    H_update = bcond_zeta(H, Z, obc_ele)
    return H_update

def check_flux_direction(z1, z2, h1, h2):
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


def reconstruct_flow_depth(H, Z, M, N):
    """
    向量化重构流深计算函数
    处理动量分量M和N的流深D_M/D_N计算
    """
    D_M = np.zeros_like(M[0, :])
    D_N = np.zeros_like(N[0, :])
    # ================= 处理M分量 (i方向通量) =================
    nstartx = 0
    nendx = Nx
    nstarty = 0
    nendy = Ny+1
    for ix in range(nstartx, nendx):
        for iy in range(nstarty, nendy):
            z1 = Z[ix, iy]
            z2 = Z[ix + 1, iy]
            h1 = H[1, ix, iy]
            h2 = H[1, ix + 1, iy]
            m0 = M[0, ix, iy]
            flux_sign, h1_update, h2_update = check_flux_direction(z1, z2, h1, h2)
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
    nendx = Nx + 1
    nstarty = 0
    nendy = Ny
    for ix in range(nstartx, nendx):
        for iy in range(nstarty, nendy):
            z1 = Z[ix, iy]
            z2 = Z[ix, iy + 1]
            h1 = H[1, ix, iy]
            h2 = H[1, ix, iy + 1]
            n0 = N[0, ix, iy]
            flux_sign, h1_update, h2_update = check_flux_direction(z1, z2, h1, h2)
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


def momentum_nonlinear_cartesian(H, Z, M, N, Wx, Wy, Pa):
    """
    Wx : wind u velocity in n+1 step, in rho point
    Wy : wind v velocity in n+1 step, in rho point
    Pa : Pressure in n+1 step, in rho point
    """
    rho_air = 1.2
    rho_water = 1025.0
    Dlimit = 1.0e-3
    phi = centerWeighting0
    D_M, D_N = reconstruct_flow_depth(H, Z, M, N)
    # get sustr svstr
    Cd = 2.5e-3
    f_cor = 1e-5
    windSpeed = np.sqrt(Wx * Wx + Wy * Wy)
    sustr = rho_air / rho_water * Cd * windSpeed * Wx
    svstr = rho_air / rho_water * Cd * windSpeed * Wy
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
            
            flux_sign, h1_update, h2_update = check_flux_direction(z1, z2, h1, h2)
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
    nendx = Nx
    nstarty = 1
    nendy = Ny - 1
    for ix in range(nstartx, nendx):
        for iy in range(nstarty, nendy):
            z1 = Z[ix, iy]
            z2 = Z[ix, iy + 1]
            h1 = H[1, ix, iy]
            h2 = H[1, ix, iy + 1]
            P1 = Pa[ix, iy]
            P2 = Pa[ix, iy + 1]
            flux_sign, h1_update, h2_update = check_flux_direction(z1, z2, h1, h2)
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
    M_update = bcond_u2D(H_update, Z, M, D_M, z_w, z_e, obc_u2d)
    N_update = bcond_v2D(H_update, Z, N, D_N, z_s, z_n, obc_v2d)
    return M_update, N_update

def bcond_zeta(H, Z, obc_ele):
    """
    :param H:  domain elevation
    :param Z:  domain depth
    :return:
    """
    #           W  S E N '
    # obc_ele = ['Cha', 'Clo', 'Gra', 'Clo']
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
            H[1, Nx - 1, iy] = (1.0 - Cx) * H[0, Nx - 1, iy] + Cx * H[0, Nx - 2, iy]
    elif obc_ele[2] == 'Cha_i':  # implicit Chapman boundary condition.
        for iy in range(Ny):
            Cx = CC1 * np.sqrt(g * (Z[Nx - 2, iy] + H[0, Nx - 2, iy]))
            cff2 = 1.0 / (1.0 + Cx)
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
            H[1, ix, 0] = (1.0 - Ce) * H[0, ix, 0] + Ce * H[0, ix, 1]
    elif obc_ele[1] == 'Cha_i':  # implicit Chapman boundary condition.
        for ix in range(Nx):
            Ce = CC2 * np.sqrt(g * (Z[ix, 1] + H[0, ix, 1]))
            cff2 = 1.0 / (1.0 + Ce)
            H[1, ix, 0] = cff2 * (H[0, ix, 0] + Ce * H[1, ix, 1])
    elif obc_ele[1] == 'Gra':
        H[1, :, 0] = H[1, :, 1]
    elif obc_ele[1] == 'Clo':
        H[1, :, 0] = 0.0
    elif obc_ele[1] == 'Rad':
        for ix in range(Nx):
            H[1, ix, 0] = H[0, ix, 0] - 2.0 * CC2 * np.sqrt(g * (Z[ix, 0] + H[0, ix, 0])) * (H[0, ix, 0] - H[0, ix, 1])
    # northern side
    if obc_ele[3] == 'Cha_e': # explicit Chapman boundary condition
        for ix in range(Nx):
            Ce = CC2 * np.sqrt(g * (Z[ix, Ny - 2] + H[0, ix, Ny - 2]))
            H[1, ix, Ny - 1] = (1.0 - Ce) * H[0, ix, Ny - 1] + Ce * H[0, ix, Ny - 2]
    elif obc_ele[3] == 'Cha_i':  # implicit Chapman boundary condition.
        for ix in range(Nx):
            Ce = CC2 * np.sqrt(g * (Z[ix, Ny - 2] + H[0, ix, Ny - 2]))
            cff2 = 1.0 / (1.0 + Ce)
            H[1, ix, Ny - 1] = cff2 * (H[0, ix, Ny - 1] + Ce * H[1, ix, Ny-2])
    elif obc_ele[3] == 'Gra':
        H[1, :, Ny - 1] = H[1, :, Ny - 2]
    elif obc_ele[3] == 'Clo':
        H[1, :, Ny - 1] = 0.0
    elif obc_ele[3] == 'Rad':
        for ix in range(Nx):
            H[1, ix, Ny - 1] = H[0, ix, Ny - 1] - 2.0 * CC2 * np.sqrt(g * (Z[ix, Ny - 1] + H[0, ix, Ny - 1])) * (H[0, ix, Ny - 1] - H[0, ix, Ny - 2])
    return H

def bcond_u2D(H, Z, M, D_M, z_w, z_e, obc_u2d):
    """
    :param H:  domain elevation
    :param Z:  depth
    :param z_e:  elevation in east bound
    :param z_w:  elevation in west bound
    :return:
    #          W  S E N '
    obc_u2d = ['Fla', 'Clo', 'Gra', 'Clo']
    """
    OneAtm = 1013.25
    rho0 = 1025
    fac = 100.0 / (g * rho0)
    Nx, Ny = H.shape[1], H.shape[2]  # 此处： L = Nx, M = Ny, Lm = Nx - 1, Mm = Ny - 1
    # Southern side
    if obc_u2d[1] == 'Fla':
        ubar_s = np.zeros((Nx - 1, 1), dtype=float)
        for ix in range(Nx-1):
            cff = dt * 0.5 / dy
            cff1 = np.sqrt(g * 0.5 * (Z[ix, 1] + H[0, ix, 1] + Z[ix + 1, 1] + H[0, ix + 1, 1]))
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
            ubar_s[ix] = M[0, ix, 0] / D_M[ix, 0] * cff - 2 * CC2 * np.sqrt(g * cff * 0.5) * cff1
            ubar_s[ix] = ubar_s[ix] / cff2
            M[1, ix, 0] = ubar_s[ix] * D_M[ix, 0]
    # northern side
    if obc_u2d[3] == 'Fla':
        ubar_n = np.zeros((Nx - 1, 1), dtype=float)
        for ix in range(Nx-1):
            cff = dt * 0.5 / dy
            cff1 = np.sqrt(g * 0.5 * (Z[ix, Ny - 2] + H[0, ix, Ny - 2] + Z[ix + 1, Ny - 2] + H[0, ix + 1, Ny - 2]))
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
            ubar_s[ix] = M[0, ix, Ny - 1] / D_M[ix, Ny - 1] * cff - 2 * CC2 * np.sqrt(g * cff * 0.5) * cff1
            ubar_s[ix] = ubar_s[ix] / cff2
            M[1, ix, Ny - 1] = ubar_n[ix] * D_M[ix, Ny - 1]
    # western side
    if obc_u2d[0] == 'Fla':
        ubar_w = np.zeros((Ny, 1), dtype=float)
        for iy in range(Ny):
            bry_pgr = -g * (H[0, 1, iy] - z_w[iy]) * 0.5 * CC1
            bry_cor = 0.0 # should add
            cff1 = 1.0 / (0.5 * (Z[0, iy] + H[0, 0, iy] + Z[1, iy] + H[0, 1, iy]))
            bry_str = 0.0 # should add surface forcing
            Cx = 1.0 / np.sqrt(g * 0.5 * (Z[0, iy] + H[0, 0, iy] + Z[1, iy] + H[0, 1, iy]))
            cff2 = Cx / dx
            bry_val = M[0, 1, iy] / D_M[1, iy] + cff2 * (bry_pgr + bry_cor + bry_str)
            Cx = np.sqrt( g * cff1 )
            ubar_w[iy] = bry_val - Cx * ( 0.5 * (H[0, 0, iy] + H[0, 1, iy]) - z_w[iy])
            M[1, 0, iy] = np.float64(ubar_w[iy]) * D_M[0, iy]
    elif obc_u2d[0] == 'Gra':
        M[1, 0, :] = M[1, 1, :]
    elif obc_u2d[0] == 'Clo':
        M[1, 0, :] = 0.0
    elif obc_u2d[0] == 'Rad':
        ubar_w = np.zeros((Ny, 1), dtype=float)
        for iy in range(Ny):
            ubar_w[iy] = -1.0 * np.sqrt(g / (Z[0, iy] + H[1, 0, iy])) * H[1, 0, iy]
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

def bcond_v2D(H, Z, N, D_N, z_s, z_n, obc_v2d):
    """
    :param H:  domain elevation
    :param Z:  depth
    :param z_s:  elevation in south bound
    :param z_n:  elevation in north bound
    :return:
    """
    OneAtm = 1013.25
    rho0 = 1025
    fac = 100.0 / (g * rho0)
    Nx, Ny = H.shape[1], H.shape[2] # 此处： L = Nx, M = Ny, Lm = Nx - 1, Mm = Ny - 1
    #          W  S E N '
    # obc_v2d = ['Fla', 'Clo', 'Gra', 'Clo']
    # western side
    if obc_v2d[0] == 'Fla':
        vbar_w = np.zeros((Ny - 1, 1), dtype=float)
        for iy in range(Ny-1):
            cff = dt * 0.5 / dx
            cff1 = np.sqrt(g * 0.5 * (Z[1, iy] + H[0, 1, iy] + Z[1, iy + 1] + H[0, 1, iy + 1]))
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
            vbar_w[iy] = N[0, 0, iy] / D_N[0, iy] * cff - 2 * CC1 * np.sqrt(g * cff * 0.5) * cff1
            vbar_w[iy] = vbar_w[iy] / cff2
            N[1, 0, iy] = vbar_w[iy] * D_N[0, iy]
    #  Eastern side
    if obc_v2d[2] == 'Fla':
        vbar_e = np.zeros((Ny - 1, 1), dtype=float)
        for iy in range(Ny - 1):
            cff = dt * 0.5 / dx
            cff1 = np.sqrt(g * 0.5 * (Z[Nx - 2, iy] + H[0, Nx - 2, iy] + Z[Nx - 2, iy + 1] + H[0, Nx - 2, iy + 1]))
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
            vbar_e[iy] = N[0, Nx - 1, iy] / D_N[Nx - 1, iy] * cff - 2 * CC1 * np.sqrt(g * cff * 0.5) * cff1
            vbar_e[iy] = vbar_e[iy] / cff2
            N[1, Nx - 1, iy] = vbar_e[iy] * D_N[Nx - 1, iy]
    # Southern side
    if obc_v2d[1] == 'Fla':
        vbar_s = np.zeros((Nx, 1), dtype=float)
        for ix in range(Nx):
            bry_pgr = -g * (H[0, ix, 0] - z_s[ix]) * 0.5 * CC2
            bry_cor = 0.0 # should add
            cff1 = 1.0 / (0.5 * (Z[ix, 0] + H[0, ix, 0] + Z[ix, 1] + H[0, ix, 1]))
            bry_str = 0.0 # should add surface forcing
            Ce = 1.0 / np.sqrt(g * 0.5 * (Z[ix, 0] + H[0, ix, 0] + Z[ix, 1] + H[0, ix, 1]))
            cff2 = Ce / dy
            bry_val = N[0, ix, 1] / D_N[ix, 1] + cff2 * (bry_pgr + bry_cor + bry_str)
            Ce = np.sqrt( g * cff1 )
            vbar_s[ix] = bry_val - Ce * ( 0.5 * (H[0, ix, 0] + H[0, ix, 1]) - z_s[ix])
            N[1, ix, 0] = vbar_s[ix] * D_N[ix, 0]
    elif obc_v2d[1] == 'Gra':
        N[1, :, 0] = N[1, :, 1]
    elif obc_v2d[1] == 'Clo':
        N[1, :, 0] = 0.0
    elif obc_v2d[1] == 'Rad':
        vbar_s = np.zeros((Nx, 1), dtype=float)
        for ix in range(Nx):
            vbar_s[ix] = -1.0 * np.sqrt(g / (Z[ix, 0] + H[1, ix, 0])) * H[1, ix, 0]
            N[1, ix, 0] = vbar_s[ix] * D_N[ix, 0]
    # northern side
    if obc_v2d[3] == 'Fla':
        vbar_n = np.zeros((Nx, 1), dtype=float)
        for ix in range(Nx):
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
    Lx = 800*1e3  # Length of domain in x-direction, in KM
    Ly = 400*1e3  # Length of domain in y-direction, in KM
    Nx = 100  # Number of grid points in x-direction
    Ny = 50 # Number of grid points in y-direction
    dx = Lx / Nx  # Grid spacing in x-direction
    dy = Ly / Ny  # Grid spacing in y-direction
    depth = 50.0
    x = np.linspace(0, Lx, Nx + 1)  # Array with x-points
    y = np.linspace(0, Ly, Ny + 1)  # Array with y-points
    #x = np.linspace(-Lx, Lx+dx*2 , 2*Nx + 1)  # For the wind
    #y = np.linspace(-Ly, Ly+dy*2 , 2*Ny + 1)  # For the wind
    X, Y = np.meshgrid(x, y)  # Meshgrid for plotting
    X = np.transpose(X)  # To get plots right
    Y = np.transpose(Y)  # To get plots right
    #          W  S E N '
    # obc_ele = ['Cha_e', 'Clo', 'Gra', 'Clo']
    # obc_u2d = ['Fla', 'Clo', 'Gra', 'Clo']
    # obc_v2d = ['Fla', 'Clo', 'Gra', 'Clo']
    obc_ele = ['Rad', 'Clo', 'Clo', 'Clo']
    obc_u2d = ['Rad', 'Clo', 'Clo', 'Clo']
    obc_v2d = ['Rad', 'Clo', 'Clo', 'Clo']
    g = 9.8
    dtRef = 0.7 * min(dx, dy) / np.sqrt(g * 100); print(dtRef)
    manning = 0.0 #2.5e-3
    dry_limit = 20
    MinWaterDepth = 0.01
    M = np.zeros((2, Nx, Ny+1))  # HU
    N = np.zeros((2, Nx+1, Ny))  # HV
    H = np.zeros((2, Nx+1, Ny+1))  # zeta
    # D_M = np.zeros((Nx, Ny+1))
    # D_N = np.zeros((Nx+1, Ny))
    Sign_M = np.zeros((Nx, Ny+1))
    Sign_N = np.zeros((Nx+1, Ny))
    Z = np.ones((Nx+1, Ny+1)) * depth # depth
    dt = 20
    NT = 1000
    centerWeighting0 = 0.9998
    FrictionDepthLimit = 5e-3
    # 临时变量
    CC1 = dt / dx
    CC2 = dt / dy
    CC3 = CC1 * g
    CC4 = CC2 * g
    # Initial condition for eta.
    # eta_n[:, :] = np.sin(4*np.pi*X/L_y) + np.sin(4*np.pi*Y/L_y)
    # eta_n = np.exp(-((X-0)**2/(2*(L_R)**2) + (Y-0)**2/(2*(L_R)**2)))
    ## eta_n = np.exp(-((X - L_x / 2.7) ** 2 / (2 * (0.05E+6) ** 2) + (Y - L_y / 4) ** 2 / (2 * (0.05E+6) ** 2)))
    #H[0, int(3*Nx/8):int(5*Nx/8),int(3*Ny/8):int(5*Ny/8)] = 1.0
    #H[0, 74:76, 74:76] = 2.0
    # eta_n[int(3*N_x/8):int(5*N_x/8), int(13*N_y/14):] = 1.0
    # Forcing
    Wr = 30
    rMax = 50*1000
    typhoon_Vec = [5,0] # !!!!!!!!!!!!!
    Wx = np.zeros((NT, x.shape[0], y.shape[0]))
    Wy = np.zeros((NT, x.shape[0], y.shape[0]))
    Pa = 1000 + Wx
    for itime in range(NT):
        typhoon_Pos = [200*1e3 + itime*dt*typhoon_Vec[0], 200*1e3 + itime*dt*typhoon_Vec[1]]
        Wx[itime], Wy[itime] = WindProfile(Wr, rMax, typhoon_Pos, typhoon_Vec, X, Y)
    
    rho_water = 1025.0
    rho_air = 1.2
    # get sustr svstr
    Cd = 2.5e-3
    Ws = windSpeed = np.sqrt(Wx * Wx + Wy * Wy)
    sustr = rho_air / rho_water * Cd * windSpeed * Wx
    svstr = rho_air / rho_water * Cd * windSpeed * Wy
    #
    # sustr = np.concatenate((sustr[:,:,0:1], sustr), 2)
    # svstr = np.concatenate((svstr[:,0:1,:], svstr), 1)
    # #sustr = np.concatenate((sustr[:,0:1,:], sustr), 1)
    # #svstr = np.concatenate((svstr[:,:,0:1], svstr), 2)
    # # lon_u = np.vstack([X[0, :] - 0.5 * (X[1, :] - X[0, :]), 0.5 * (X[:-1, :] + X[1:, :]), X[-1, :] + 0.5 * (X[-1, :] - X[-2, :])])
    # # lat_u = np.vstack([Y[0, :] - 0.5 * (Y[1, :] - Y[0, :]), 0.5 * (Y[:-1, :] + Y[1:, :]), Y[-1, :] + 0.5 * (Y[-1, :] - Y[-2, :])])
    # # lon_v = np.hstack([X[:, 0:1] - 0.5 * (X[:, 1:2] - X[:, 0:1]), 0.5 * (X[:, :-1] + X[:, 1:]), X[:, -1:] + 0.5 * (X[:, -1:] - X[:, -2:-1])])
    # # lat_v = np.hstack([Y[:, 0:1] - 0.5 * (Y[:, 1:2] - Y[:, 0:1]), 0.5 * (Y[:, :-1] + Y[:, 1:]), Y[:, -1:] + 0.5 * (Y[:, -1:] - Y[:, -2:-1])])
    # lon_u = np.hstack([X[:, 0:1] - 0.5 * (X[:, 1:2] - X[:, 0:1]), 0.5 * (X[:, :-1] + X[:, 1:]), X[:, -1:] + 0.5 * (X[:, -1:] - X[:, -2:-1])])
    # lat_u = np.hstack([Y[:, 0:1] - 0.5 * (Y[:, 1:2] - Y[:, 0:1]), 0.5 * (Y[:, :-1] + Y[:, 1:]), Y[:, -1:] + 0.5 * (Y[:, -1:] - Y[:, -2:-1])])
    # lon_v = np.vstack([X[0, :] - 0.5 * (X[1, :] - X[0, :]), 0.5 * (X[:-1, :] + X[1:, :]), X[-1, :] + 0.5 * (X[-1, :] - X[-2, :])])
    # lat_v = np.vstack([Y[0, :] - 0.5 * (Y[1, :] - Y[0, :]), 0.5 * (Y[:-1, :] + Y[1:, :]), Y[-1, :] + 0.5 * (Y[-1, :] - Y[-2, :])])
    # time = (  np.arange(0, (NT)*dt,dt, dtype=np.float64) )/86400.
    # ds_out = xr.Dataset(
    #         {'sustr': (['sms_time','lat_u', 'lon_u'], sustr.swapaxes(1,2)*1e3),  #for ROMS
    #          'svstr': (['sms_time','lat_v', 'lon_v'], svstr.swapaxes(1,2)*1e3),
    #          'lon_u': (['lat_u','lon_u'], lon_u.transpose()),
    #          'lat_u': (['lat_u','lon_u'], lat_u.transpose()),
    #          'lon_v': (['lat_v','lon_v'], lon_v.transpose()),
    #          'lat_v': (['lat_v','lon_v'], lat_v.transpose())},
    #         coords={
    #             'sms_time': time}   )
    # #写入字段
    # ds_out['sustr'].attrs['long_name'] = "sustr"
    # ds_out['sustr'].attrs['units'] = "N m-2"
    # ds_out['sustr'].attrs['coordinates'] = "lat_u lon_u"
    # ds_out['sustr'].attrs['time'] = "sms_time"
    # ds_out['svstr'].attrs['long_name'] = "svstr"
    # ds_out['svstr'].attrs['units'] = "N m-2"
    # ds_out['svstr'].attrs['coordinates'] = "lat_v lon_v"
    # ds_out['svstr'].attrs['time'] = "sms_time" 
    
    # ds_out['sms_time'].attrs['units'] = "days since 0001-01-01 00:00:00"
    
    # ds_out.to_netcdf('WindStrMove'+str(Wr)+str(int(rMax/1000))+'.nc')

    #pcolor(X,Y,(Wx**2+Wy**2)**0.5);colorbar()
    eta_list = list()
    u_list = list()
    v_list = list()
    anim_interval = 1
    for itime in range(NT):
        print(f"nt = {itime} / {NT}")
        H_update = mass_cartesian(H, Z, M, N)
        M_update, N_update = momentum_nonlinear_cartesian(H_update, Z, M, N, Wx[itime],Wy[itime],Pa[itime]*0)

        H = H_update.copy()
        M = M_update.copy()
        N = N_update.copy()
        H[0, :, :] = H[1, :, :]
        M[0, :, :] = M[1, :, :]
        N[0, :, :] = N[1, :, :]
        eta_list.append(H_update[1, :])
        u_list.append(M_update[1, :])
        v_list.append(N_update[1, :])
        
        
        
        pcolor(X, Y, eta_list[-1], vmin=-.2,vmax=.2, cmap=plt.cm.RdBu_r)
        colorbar()
        xlabel("x [m]", fontname="serif", fontsize=12)
        ylabel("y [m]", fontname="serif", fontsize=12)
        title("Stage at an instant in time: " + f"{itime*dt}" + " second")
        axis('equal')
        #fig.savefig("ele_Rad" + f"{itime}" + "s.jpg", format='jpg', dpi=300, bbox_inches='tight')
        show()
        
    
    
    # ds_out = xr.Dataset(
    #         {'eta': (['time','lat', 'lon'], np.array(eta_list))},
    #         coords={
    #             'lat': x,
    #             'lon': y,
    #             'time': np.arange(itime+1)}  )
    # ds_out.to_netcdf('out.circulation.nc')
    
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
        #quiver(X[::3,::3],Y[::3,::3],(Wx/Ws)[ind2plot,::3,::3],(Wy/Ws)[ind2plot,::3,::3], scale=70)
        quiver(X[::3,::3],Y[::3,::3],(Wx/Ws)[ind2plot,::3,::3],(Wy/Ws)[ind2plot,::3,::3], scale=70)
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