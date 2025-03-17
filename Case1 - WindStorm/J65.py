import numpy as np
from matplotlib.pyplot import *

def max_wind_radius(P0, Rk):
    dp = P0 - 900
    return (Rk - 0.4 * dp + 0.01 * dp * dp) * 1000.0

def max_wind_speed(P0, P_base):
    return 3.029 * np.power(P_base - P0, 0.644)

def get_distance(typhoon_Pos, xx, yy):
    return np.sqrt((xx - typhoon_Pos[0]) * (xx - typhoon_Pos[0]) + (yy - typhoon_Pos[1]) * (yy - typhoon_Pos[1]))

def PrsProfile(P0, P_base, rMax, typhoon_Pos, xx, yy):
    r = get_distance(typhoon_Pos, xx, yy)
    Prs = np.zeros_like(r)
    P_in = P0 + 0.25 * (P_base - P0) * np.power(r / rMax, 3)
    P_out = P_base - 0.75 * (P_base - P0) * rMax / r
    Prs[np.where(r <=rMax)] = P_in [np.where(r <=rMax)]
    Prs[np.where(r > rMax)] = P_out[np.where(r > rMax)]
    return  Prs

def WindProfile(Wr, rMax, typhoon_Pos, typhoon_Vec, xx, yy):
    """
    Wr: max wind speed
    rMax: max wind radius
    r:  distance to typhoon center
    typhoon_Pos: [xc, yc], Position of Typhoon
    P0: Pressure of Typhoon center
    typhoon_Vec: move vec of typhoon [x,y] in 
    xx: position of mesh grid
    yy: position of mesh grid
    """
    r = get_distance(typhoon_Pos, xx, yy) + 1e-6
    sita = np.zeros_like(r) + 25.0
    Wx = np.zeros_like(r)
    Wy = np.zeros_like(r)
    sita[r <= rMax] = 10.0
    nx, ny = r.shape[0], r.shape[1]
    beta = 0.50
    for ix in range(nx):
        for iy in range(ny):
            if r[ix, iy] <= 1.2 * rMax and r[ix, iy] > 1.0 * rMax:
                sita[ix, iy] = 10.0 + (1.2 * rMax - r[ix, iy]) * (25.0 - 10.0) / (1.2 * rMax - rMax)
    A = -1.0 * ((xx - typhoon_Pos[0]) * np.sin(sita * np.pi/180) + (yy - typhoon_Pos[1]) * np.cos(sita * np.pi/180))
    B =  1.0 * ((xx - typhoon_Pos[0]) * np.cos(sita * np.pi/180) - (yy - typhoon_Pos[1]) * np.sin(sita * np.pi/180))
    wx_in = (r / (r + rMax)) * typhoon_Vec[0] + Wr * np.power(r / rMax, 1.5) * A / r
    wy_in = (r / (r + rMax)) * typhoon_Vec[1] + Wr * np.power(r / rMax, 1.5) * B / r

    wx_out = (rMax / (r + rMax)) * typhoon_Vec[0] + Wr * np.power(rMax / r, beta) * A / r
    wy_out = (rMax / (r + rMax)) * typhoon_Vec[1] + Wr * np.power(rMax / r, beta) * B / r

    Wx[np.where(r <= rMax)] = wx_in [np.where(r <= rMax)]
    Wx[np.where(r >  rMax)] = wx_out[np.where(r >  rMax)]
    Wy[np.where(r <= rMax)] = wy_in [np.where(r <= rMax)]
    Wy[np.where(r >  rMax)] = wy_out[np.where(r >  rMax)]

    return Wx, Wy


