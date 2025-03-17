def mass_cartesian(H, Z, M, N, params):
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
    mask_dry = Z <= -params.dry_limit
    H[:, mask_dry] = 0.0
    nstartx = 1
    nendx = params.Nx
    nstarty = 1
    nendy = params.Ny
    for ix in range(nstartx, nendx):
        for iy in range(nstarty, nendy):
            if Z[ix, iy] <= - params.dry_limit:
                H[1, ix, iy] = 0.0
                continue
            m1 = M0[ix-1, iy]
            m2 = M0[ix, iy]
            n1 = N0[ix, iy - 1]
            n2 = N0[ix, iy]
            H[1, ix, iy] = H[0, ix, iy] - params.CC1 *(m2 - m1) - params.CC2 * (n2 - n1)

            if (Z[ix, iy] < params.dry_limit) and (Z[ix, iy] > -params.dry_limit):
                # Wet -> Dry
                if ((Z[ix, iy] + H[0, ix, iy] > 0.0) and
                        (H[1, ix, iy] - H[0, ix, iy] < 0.0) and
                        (Z[ix, iy] + H[1, ix, iy] <= params.MinWaterDepth)):

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

    H_update = bcond_zeta(H, Z, params)
    return H_update