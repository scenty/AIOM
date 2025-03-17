import numpy as np
#import vis_tools
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import *

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
    nstartx = 0
    nendx = Nx
    nstarty = 0
    nendy = Ny
    for ix in range(nstartx, nendx):
        for iy in range(nstarty, nendy):
            if Z[ix, iy] <= - dry_limit:
                H[1, ix, iy] = 0.0
                continue
            if ix == 0:
                m1 = 0.0
            else:
                m1 = M0[ix-1, iy]
            if ix == Nx - 1:
                m2 = 0.0
            else:
                m2 = M0[ix, iy]
            if iy == 0:
                n1 = 0.0
            else:
                n1 = N0[ix, iy - 1]
            if iy == Ny - 1:
                n2 = 0.0
            else:
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
    return H

def check_flux_direction(z1, z2, h1, h2):
    if z1 <= -dry_limit or z2 <= -dry_limit:
        return 999, h1, h2
    if (z1 + h1) <=0.0 or (z2 + h2) <= 0.0:
        if (z1 + h1) <= 0.0 and (z2 + h2) <= 0.0:
            return 999, h1, h2
        elif (z1 > z2) and (z1 + h1 > 0.0) and (z2 + h1 <=0.0):
            return 999, h1, h2
        elif (z2 > z1) and (z2 + h2 > 0.0) and (z1 + h2 <=0.0):
            return 999,  h1, h2
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
    nendx = Nx-1
    nstarty = 0
    nendy = Ny
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
                D_M[ix, iy] = 0.5 * (z1 + z2) + max(h1_update, h2_update)
    # ================= 处理N分量 (i方向通量) =================
    nstartx = 0
    nendx = Nx
    nstarty = 0
    nendy = Ny - 1
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
                D_N[ix, iy] = 0.5 * (z1 + z2) + max(h1_update, h2_update)

    return D_M, D_N


def momentum_nonlinear_cartesian(H, Z, M, N):
    """

    """
    Dlimit = 1.0e-3
    phi = centerWeighting0
    D_M, D_N = reconstruct_flow_depth(H, Z, M, N)
    # ================= 处理M分量 (i方向通量) =================
    nstartx = 0
    nendx = Nx - 1
    nstarty = 0
    nendy = Ny
    for ix in range(nstartx, nendx):
        for iy in range(nstarty, nendy):
            z1 = Z[ix, iy]
            z2 = Z[ix + 1, iy]
            h1 = H[1, ix, iy]
            h2 = H[1, ix + 1, iy]
            flux_sign, h1_update, h2_update = check_flux_direction(z1, z2, h1, h2)
            if flux_sign == 999:
                M[1, ix, iy] = 0.0
            else:
                # !///solve NSWE///!
                MU1 = MU2 = NU1 = NU2 = 0.0
                m0 = M[0, ix, iy]
                D0 = D_M[ix, iy]
                if ix == 0:
                    m1, D1 = 0.0, D0
                else:
                    m1, D1 = M[0, ix - 1, iy], D_M[ix - 1, iy]
                if ix == Nx - 2:
                    m2, D2 = 0.0, D0
                else:
                    m2, D2 = M[0, ix + 1, iy], D_M[ix + 1, iy]

                if iy == 0:
                    m3, D3, n3, n4 = 0.0, D0, 0.0, 0.0
                else:
                    m3, D3 = M[0, ix, iy - 1], D_M[ix, iy - 1]
                    n3, n4 = N[0, ix, iy - 1], N[0, ix + 1, iy - 1]

                if iy == Ny - 1:
                    m5, D5, n0, n2 = 0.0, D0, 0.0, 0.0
                else:
                    m5, D5 = M[0, ix, iy + 1], D_M[ix, iy + 1]
                    n0, n2 = N[0, ix, iy], N[0, ix+ 1, iy]
                # !/// extra terms n5, n6, n7, n8 ///!
                if iy <= 1:
                    n7, n8 = 0.0, 0.0
                else:
                    n7, n8 = N[0, ix, iy - 2], N[0, ix+ 1, iy - 2]

                if iy >= Ny - 2:
                    n5, n6 = 0.0, 0.0
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
                # Flux-centered
                phi = 1.0 - CC1 * min(np.abs(m0 / max(D0, MinWaterDepth)), np.sqrt(g * D0))
                M[1, ix, iy] = (phi * m0 + 0.5 * (1.0 - phi) * (m1 + m2)
                            - CC3 * D0 * (h2_update - h1_update)
                            - CC1 * (MU2 - MU1)
                            - CC2 * (NU2 - NU1)  )
                # Bottom friction
                if D0 > FrictionDepthLimit:
                    MM = m0 ** 2
                    NN = (0.25 * (n0 + n2 + n3 + n4)) ** 2
                    Cf = manning
                    Fx = g * Cf ** 2 / (D0 ** 2.33) * np.sqrt(MM + NN) * m0
                    if abs(dt * Fx) > abs(m0):
                        Fx = m0 / dt
                    M[1, ix, iy] -= dt * Fx
                # Check direction of M
                if flux_sign != 0 and M[0, ix, iy] * flux_sign < 0.0:
                    M[1, ix, iy] = 0.0
    # ================= 处理N分量 (i方向通量) =================
    nstartx = 0
    nendx = Nx
    nstarty = 0
    nendy = Ny - 1
    for ix in range(nstartx, nendx):
        for iy in range(nstarty, nendy):
            z1 = Z[ix, iy]
            z2 = Z[ix, iy + 1]
            h1 = H[1, ix, iy]
            h2 = H[1, ix, iy + 1]
            flux_sign, h1_update, h2_update = check_flux_direction(z1, z2, h1, h2)
            if flux_sign == 999:
                N[1, ix, iy] = 0.0
            else:
                # !///solve NSWE///!
                NV1, NV2, MV1, MV2 = 0.0, 0.0, 0.0, 0.0
                n0 = N[0, ix, iy]
                D0 = D_N[ix, iy]
                if iy == 0:
                    n3, D3 = 0.0, D0
                else:
                    n3, D3 = N[0, ix, iy - 1], D_N[ix, iy - 1]
                if iy == Ny - 2:
                    n5, D5 = 0.0, D0
                else:
                    n5, D5 = N[0, ix, iy + 1], D_N[ix, iy + 1]
                if ix == 1:
                    n1, D1, m1, m4 = 0.0, D0, 0.0, 0.0
                else:
                    n1, D1 = N[0, ix - 1, iy], D_N[ix - 1, iy]
                    m1, m4 = M[0, ix - 1, iy], M[0, ix - 1, iy + 1]

                if ix == Nx - 1:
                    n2, D2, m0, m5 = 0.0, D0, 0.0, 0.0
                else:
                    n2, D2 = N[0, ix + 1, iy], D_N[ix + 1, iy]
                    m0, m5 = M[0, ix, iy], M[0, ix, iy + 1]
                #!/// extra terms m2, m6, m7, m8///!
                if ix <= 1:
                    m7, m8 = 0.0, 0.0
                else:
                    m7, m8 = M[0, ix - 2, iy], M[0, ix - 2, iy + 1]

                if ix >= Nx - 2:
                    m2, m6 = 0.0, 0.0
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

                phi = 1.0 - CC2 * min(np.abs(n0 / max(D0, MinWaterDepth)),
                                      np.sqrt(g * D0))

                N[1, ix, iy] = (phi * n0 + 0.5 * (1.0 - phi) * (n3 + n5)
                                                  - CC4 * D0 * (h2_update - h1_update)
                                                  - CC2 * (NV2 - NV1)
                                                  - CC1 * (MV2 - MV1))
                if D0 > FrictionDepthLimit:
                    MM, NN = (0.25 * (m0 + m1 + m4 + m5)) ** 2, n0 ** 2
                    Cf = manning
                    Fy = g * Cf ** 2 / (D0 ** 2.33) * np.sqrt(MM + NN) * n0
                    if np.abs(dt * Fy) > abs(n0):
                        Fy = n0 / dt
                    N[1, ix, iy] -= dt * Fy
                if flux_sign != 0 and N[1, ix, iy] * flux_sign < 0.0:
                    N[1, ix, iy] = 0.0
    return M, N

def bed_elevation(x, y):
    nx, ny = x.shape
    z = np.zeros_like(x)
    for ix in range(nx):
        for iy in range(ny):
            r = np.sqrt(x[ix, iy] * x[ix, iy] + y[ix, iy] * y[ix, iy])
            z[ix, iy] = -D0 * (1. - r * r / L / L)
    return z

def stage_init(x, y):
    z = bed_elevation(x, y)
    nx, ny = x.shape
    w = np.zeros_like(x)
    for ix in range(nx):
        for iy in range(ny):
            r = np.sqrt(x[ix, iy] * x[ix, iy] + y[ix, iy] * y[ix, iy])
            w[ix, iy] = D0 * ((np.sqrt(1 - A * A)) / (1. - A * np.cos(omega * t))
                         - 1. - r * r / L / L * ((1. - A * A) / ((1. - A * np.cos(omega * t)) ** 2) - 1.))
    if w[ix, iy] < z[ix, iy]:
        w[ix, iy] = z[ix, iy]
    return w

def analytic_sol(x,y,t,D0=1000.,L=2500.,R0=2000.):
    A = (L**4 - R0**4)/(L**4 + R0**4)
    omega = 2./L*np.sqrt(2.*g*D0)
    z = bed_elevation(x,y)
    nx, ny = x.shape
    w = np.zeros_like(x)
    u = np.zeros_like(x)
    h = np.zeros_like(x)
    for ix in range(nx):
        for iy in range(ny):
            r = np.sqrt(x[ix, iy]*x[ix, iy] + y[ix, iy]*y[ix, iy])
            w[ix, iy] = D0*((np.sqrt(1-A*A))/(1.-A*np.cos(omega*t))
                -1.-r*r/L/L*((1.-A*A)/((1.-A*np.cos(omega*t))**2)-1.))
            u[ix, iy] = 0.5*omega*r*A*np.sin(omega*t) / (1.0-A*np.cos(omega*t))
            if x[ix, iy] < 0.0:
                u[ix, iy] = -u[ix, iy]
            h[ix, iy] = w[ix, iy] - z[ix, iy]
            if w[ix, iy] < z[ix, iy]:
                w[ix, iy] = z[ix, iy]
                u[ix, iy] = 0.0
                h[ix, iy] = 0.0
    return w,u,h

def plot_stage(X, Y, var2plot, time2plot):
    ind2read = time2plot
    w2, u2, h2 = analytic_sol(X, Y, time2plot * dt)
    x_ind = 37
    fig, ax = plt.subplots(1, 1)
    plt.plot(X[x_ind, :], var2plot[ind2read][x_ind, :], 'b.', label='numerical stage')
    plt.plot(X[x_ind, :], -Z[x_ind, :], 'k-', label='bed elevation')
    plt.plot(X[x_ind, :], w2[x_ind, :], 'r-', label='analytical stage')
    # f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    plt.title("Stage at an instant in time: " + f"{time2plot*dt}"+ " second")
    plt.legend(loc='best')
    plt.xlabel('Xposition')
    plt.ylabel('Stage')
    fig.savefig("cross_section_stage" + f"{time2plot*dt}" + ".jpg",
                format='jpg', dpi=300, bbox_inches='tight')
    plt.close()

def plot_Xmom(X, Y, var2plot, time2plot):
    ind2read = time2plot
    w2, u2, h2 = analytic_sol(X, Y, time2plot * dt)
    x_ind = 37
    fig, ax = plt.subplots(1, 1)
    plt.plot(X[x_ind, :-1], var2plot[ind2read][:, x_ind], 'b.', label='numerical stage')
    plt.plot(X[x_ind, :], u2[x_ind, :] * h2[x_ind, :], 'r-', label='analytical stage')
    #plt.plot(X[x_ind, :], u2[x_ind, :], 'r-', label='analytical stage')
    # f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    plt.title("Xmomentum at an instant in time: " + f"{time2plot*dt}"+ " second")
    plt.legend(loc='best')
    plt.xlabel('Xposition')
    plt.ylabel('Stage')
    fig.savefig("cross_section_Xmomentum" + f"{time2plot*dt}" + ".jpg",
                format='jpg', dpi=300, bbox_inches='tight')
    plt.close()

def plot_stage_Point(var2plot):
    # Parameters
    time2plot = len(var2plot)
    time = np.arange(time2plot) * dt
    D0 = 1000.
    L = 2500.
    R0 = 2000.
    g = 9.81
    A = (L ** 4 - R0 ** 4) / (L ** 4 + R0 ** 4)
    omega = 2. / L * np.sqrt(2. * g * D0)
    T = np.pi / omega
    # Free surface over time at the origin of spatial domain.
    w = D0 * ((np.sqrt(1. - A * A)) / (1. - A * np.cos(omega * time)) - 1.)
    x_ind = 37
    y_ind = 37
    var2plot = [_[x_ind, y_ind] for _ in var2plot]
    fig, ax = plt.subplots(1, 1)
    plt.plot(time, var2plot, 'b.', label='numerical')
    plt.plot(time, w, 'r-', label='analytical')
    # f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    plt.title("Stage at the centre of the paraboloid basin over time")
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Stage')
    fig.savefig("Stage_origin.jpg", format='jpg', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':

    Lx = 8000.0
    Ly = 8000.0
    Nx = 75  # Number of grid points in x-direction
    Ny = 75  # Number of grid points in y-direction
    dx = Lx / (Nx - 1)  # Grid spacing in x-direction
    dy = Ly / (Ny - 1)  # Grid spacing in y-direction
    x = np.linspace(-Lx / 2, Lx / 2, Nx)  # Array with x-points
    y = np.linspace(-Ly / 2, Ly / 2, Ny)  # Array with y-points
    X, Y = np.meshgrid(x, y)  # Meshgrid for plotting
    X0 = np.transpose(X)  # To get plots right
    Y0 = np.transpose(Y)  # To get plots right

    g = 9.8
    dtRef = 0.1 * min(dx, dy) / np.sqrt(g * 100)
    manning = 0 #2.5e-3
    dry_limit = 1e5
    MinWaterDepth = 0.001
    M = np.zeros((2, Nx-1, Ny))  # HU
    N = np.zeros((2, Nx, Ny-1))  # HV
    H = np.zeros((2, Nx, Ny))  # zeta
    D_M = np.zeros((Nx-1, Ny))
    D_N = np.zeros((Nx, Ny-1))
    Sign_M = np.zeros((Nx-1, Ny))
    Sign_N = np.zeros((Nx, Ny-1))
    #Z = np.ones((Nx, Ny)) * 100 # depth
    dt = 0.1
    NT = 2000
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
    # -------------------------------------------------------------------------------
    # Initial conditions
    # -------------------------------------------------------------------------------
    t = 0.0
    D0 = 1000.
    L = 2500.
    R0 = 2000.
    A = (L ** 4 - R0 ** 4) / (L ** 4 + R0 ** 4)
    omega = 2. / L * np.sqrt(2. * g * D0)
    T = np.pi / omega
    Z = -bed_elevation(X, Y)
    H[0, :] = stage_init(X, Y)
    plot(Z[35]);plot(H[0,35]);legend(('Z','H_init'))
    eta_list = list()
    u_list = list()
    v_list = list()
    anim_interval = 1
    for itime in range(NT):
        print(f"nt = {itime} / {NT}")
        H_update = mass_cartesian(H, Z, M, N)
        M_update, N_update = momentum_nonlinear_cartesian(H_update, Z, M, N)
        H = H_update.copy()
        M = M_update.copy()
        N = N_update.copy()
        H[0, :, :] = H[1, :, :]
        M[0, :, :] = M[1, :, :]
        N[0, :, :] = N[1, :, :]
        plot(-Z[35]);plot(H[0,35]);legend(('Z','H_init'));
        title(f"{itime*dt:.2f}s")
        show()
        eta_list.append(H_update[1, :])
        u_list.append(M_update[1, :])
        v_list.append(N_update[1, :])

    # for itime in range(5, 20):
    #     #plot_stage(X, Y, eta_list, itime*100) # cause dt = 0.1 , 50s 60s ~100s
    #     plot_Xmom(X, Y, u_list, itime*100)
    plot_stage_Point(eta_list)
    # fig, ax = plt.subplots(1, 1)
    # # plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 17)
    # plt.xlabel("x [m]", fontname="serif", fontsize=12)
    # plt.ylabel("y [m]", fontname="serif", fontsize=12)
    # #var2plot = eta_list[-1]
    # var2plot = Z + H[0, :]
    # pmesh = plt.pcolormesh(X, Y, var2plot, vmin=-0.7 * np.abs(var2plot[int(len(var2plot) / 2)]).max(),
    #                        vmax=np.abs(var2plot[int(len(var2plot) / 2)]).max(), cmap=plt.cm.RdBu_r)
    # plt.colorbar(pmesh, orientation="vertical")
    # fig.savefig("ele_ini.jpg", format='jpg', dpi=300, bbox_inches='tight')
    # plt.close()

    # fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    # plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname="serif", fontsize=19)
    # plt.xlabel("x [km]", fontname="serif", fontsize=16)
    # plt.ylabel("y [km]", fontname="serif", fontsize=16)
    # q_int = 5
    # Q = ax.quiver(X[::q_int, ::q_int] / 1000.0, Y[::q_int, ::q_int] / 1000.0, u_list[-1][::q_int, ::q_int]/100,
    #               v_list[-1][::q_int, ::q_int]/100,
    #               scale=0.2, scale_units='inches')
    # fig.savefig("vec.jpg", format='jpg', dpi=300, bbox_inches='tight')
    # plt.close()
    #eta_anim = vis_tools.eta_animation(X, Y, eta_list, anim_interval*dt, "eta")