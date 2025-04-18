import numpy as np
import torch
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

def generate_wind(X,Y,params,output = False):
    NT = params.NT
    Wx = np.ones((NT,X.shape[0],X.shape[1]))*0
    Wy = np.ones((NT,X.shape[0],X.shape[1]))*0
    #Pa = Wx * 0 + 1000
    for itime in range(params.NT):
        typhoon_Pos = params.typhoon_Pos[itime]
        Wx[itime], Wy[itime] = WindProfile(params.Wr, params.rMax, typhoon_Pos, params.typhoon_Vec, X, Y)
    
    Wx = torch.from_numpy(Wx).to(dtype=X.dtype, device=X.device)
    Wy = torch.from_numpy(Wy).to(dtype=X.dtype, device=X.device)
    
    Pa = 1000 + Wx*0
    Ws = windSpeed = torch.sqrt(Wx * Wx + Wy * Wy)
    sustr = params.rho_air / params.rho_water * params.Cd * windSpeed * Wx
    svstr = params.rho_air / params.rho_water * params.Cd * windSpeed * Wy
    
    if output:
        sustr = np.concatenate((sustr[:,:,0:1], sustr), 2)
        svstr = np.concatenate((svstr[:,0:1,:], svstr), 1)
        lon_u = np.hstack([X[:, 0:1] - 0.5 * (X[:, 1:2] - X[:, 0:1]), 0.5 * (X[:, :-1] + X[:, 1:]), X[:, -1:] + 0.5 * (X[:, -1:] - X[:, -2:-1])])
        lat_u = np.hstack([Y[:, 0:1] - 0.5 * (Y[:, 1:2] - Y[:, 0:1]), 0.5 * (Y[:, :-1] + Y[:, 1:]), Y[:, -1:] + 0.5 * (Y[:, -1:] - Y[:, -2:-1])])
        lon_v = np.vstack([X[0, :] - 0.5 * (X[1, :] - X[0, :]), 0.5 * (X[:-1, :] + X[1:, :]), X[-1, :] + 0.5 * (X[-1, :] - X[-2, :])])
        lat_v = np.vstack([Y[0, :] - 0.5 * (Y[1, :] - Y[0, :]), 0.5 * (Y[:-1, :] + Y[1:, :]), Y[-1, :] + 0.5 * (Y[-1, :] - Y[-2, :])])
        time = (  np.arange(0, (NT)*dt,dt, dtype=np.float64) )/86400.
        ds_out = xr.Dataset(
                {'sustr': (['sms_time','lat_u', 'lon_u'], sustr.swapaxes(1,2)*1e3),  #for ROMS
                 'svstr': (['sms_time','lat_v', 'lon_v'], svstr.swapaxes(1,2)*1e3),
                 'lon_u': (['lat_u','lon_u'], lon_u.transpose()),
                 'lat_u': (['lat_u','lon_u'], lat_u.transpose()),
                 'lon_v': (['lat_v','lon_v'], lon_v.transpose()),
                 'lat_v': (['lat_v','lon_v'], lat_v.transpose())},
                coords={
                    'sms_time': time}   )
        #写入字段
        ds_out['sustr'].attrs['long_name'] = "sustr"
        ds_out['sustr'].attrs['units'] = "N m-2"
        ds_out['sustr'].attrs['coordinates'] = "lat_u lon_u"
        ds_out['sustr'].attrs['time'] = "sms_time"
        ds_out['svstr'].attrs['long_name'] = "svstr"
        ds_out['svstr'].attrs['units'] = "N m-2"
        ds_out['svstr'].attrs['coordinates'] = "lat_v lon_v"
        ds_out['svstr'].attrs['time'] = "sms_time" 
        
        ds_out['sms_time'].attrs['units'] = "days since 0001-01-01 00:00:00"
        
        ds_out.to_netcdf('WindStrMove'+str(Wr)+str(int(rMax/1000))+'.nc')

    #pcolor(X,Y,(Wx**2+Wy**2)**0.5);colorbar()
    return Wx, Wy, Pa
