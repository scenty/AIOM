# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:25:23 2025

@author: ASUS
"""

import torch
import xarray as xr

def max_wind_radius(P0, Rk):
    dp = P0 - 900
    return (Rk - 0.4 * dp + 0.01 * dp * dp) * 1000.0

def max_wind_speed(P0, P_base):
    return 3.029 * torch.pow(P_base - P0, 0.644)

def get_distance(typhoon_Pos, xx, yy):
    # Ensure that all tensors are on the same device
    typhoon_Pos = typhoon_Pos.to(xx.device)  # Move typhoon_Pos to the same device as xx, yy
    return torch.sqrt((xx - typhoon_Pos[0]) ** 2 + (yy - typhoon_Pos[1]) ** 2)

def PrsProfile(P0, P_base, rMax, typhoon_Pos, xx, yy):
    r = get_distance(typhoon_Pos, xx, yy)
    Prs = torch.zeros_like(r)
    P_in = P0 + 0.25 * (P_base - P0) * torch.pow(r / rMax, 3)
    P_out = P_base - 0.75 * (P_base - P0) * rMax / r
    Prs[r <= rMax] = P_in[r <= rMax]
    Prs[r > rMax] = P_out[r > rMax]
    return Prs

def WindProfile(Wr, rMax, typhoon_Pos, typhoon_Vec, xx, yy):
    """
    Wr: max wind speed
    rMax: max wind radius
    r: distance to typhoon center
    typhoon_Pos: [xc, yc], Position of Typhoon
    P0: Pressure of Typhoon center
    typhoon_Vec: move vec of typhoon [x,y]
    xx: position of mesh grid
    yy: position of mesh grid
    """
    r = get_distance(typhoon_Pos, xx, yy) + 1e-6
    sita = torch.full_like(r, 25.0)
    Wx = torch.zeros_like(r)
    Wy = torch.zeros_like(r)
    
    sita[r <= rMax] = 10.0
    nx, ny = r.shape[0], r.shape[1]
    beta = 0.50
    for ix in range(nx):
        for iy in range(ny):
            if r[ix, iy] <= 1.2 * rMax and r[ix, iy] > 1.0 * rMax:
                sita[ix, iy] = 10.0 + (1.2 * rMax - r[ix, iy]) * (25.0 - 10.0) / (1.2 * rMax - rMax)

    A = -1.0 * ((xx - typhoon_Pos[0]) * torch.sin(sita * torch.acos(torch.tensor(-1.0)) / 180) + (yy - typhoon_Pos[1]) * torch.cos(sita * torch.acos(torch.tensor(-1.0)) / 180))
    B = 1.0 * ((xx - typhoon_Pos[0]) * torch.cos(sita * torch.acos(torch.tensor(-1.0)) / 180) - (yy - typhoon_Pos[1]) * torch.sin(sita * torch.acos(torch.tensor(-1.0)) / 180))
    
    wx_in = (r / (r + rMax)) * typhoon_Vec[0] + Wr * torch.pow(r / rMax, 1.5) * A / r
    wy_in = (r / (r + rMax)) * typhoon_Vec[1] + Wr * torch.pow(r / rMax, 1.5) * B / r

    wx_out = (rMax / (r + rMax)) * typhoon_Vec[0] + Wr * torch.pow(rMax / r, beta) * A / r
    wy_out = (rMax / (r + rMax)) * typhoon_Vec[1] + Wr * torch.pow(rMax / r, beta) * B / r

    Wx[r <= rMax] = wx_in[r <= rMax]
    Wx[r > rMax] = wx_out[r > rMax]
    Wy[r <= rMax] = wy_in[r <= rMax]
    Wy[r > rMax] = wy_out[r > rMax]

    return Wx, Wy

def generate_wind(X, Y, params, output=False):
    NT = params.NT
    Wx = torch.zeros((NT, X.shape[0], X.shape[1]), device=X.device)
    Wy = torch.zeros((NT, X.shape[0], X.shape[1]), device=X.device)
    
    for itime in range(NT):
        typhoon_Pos = params.typhoon_Pos[itime]
        Wx[itime], Wy[itime] = WindProfile(params.Wr, params.rMax, typhoon_Pos, params.typhoon_Vec, X, Y)
    
    # Convert Wx and Wy to the same device as X
    Pa = 1000 + Wx * 0
    Ws = torch.sqrt(Wx ** 2 + Wy ** 2)
    
    sustr = (params.rho_air / params.rho_water) * params.Cd * Ws * Wx
    svstr = (params.rho_air / params.rho_water) * params.Cd * Ws * Wy
    
    if output:
        sustr = torch.cat((sustr[:, :, 0:1], sustr), dim=2)
        svstr = torch.cat((svstr[:, 0:1, :], svstr), dim=1)
        
        lon_u = torch.cat([X[:, 0:1] - 0.5 * (X[:, 1:2] - X[:, 0:1]), 0.5 * (X[:, :-1] + X[:, 1:]), X[:, -1:] + 0.5 * (X[:, -1:] - X[:, -2:-1])], dim=1)
        lat_u = torch.cat([Y[:, 0:1] - 0.5 * (Y[:, 1:2] - Y[:, 0:1]), 0.5 * (Y[:, :-1] + Y[:, 1:]), Y[:, -1:] + 0.5 * (Y[:, -1:] - Y[:, -2:-1])], dim=1)
        lon_v = torch.cat([X[0, :] - 0.5 * (X[1, :] - X[0, :]), 0.5 * (X[:-1, :] + X[1:, :]), X[-1, :] + 0.5 * (X[-1, :] - X[-2, :])], dim=0)
        lat_v = torch.cat([Y[0, :] - 0.5 * (Y[1, :] - Y[0, :]), 0.5 * (Y[:-1, :] + Y[1:, :]), Y[-1, :] + 0.5 * (Y[-1, :] - Y[-2, :])], dim=0)
        
        time = (torch.arange(0, NT * params.dt, params.dt, dtype=torch.float64) ) / 86400.
        
        ds_out = xr.Dataset(
            {'sustr': (['sms_time', 'lat_u', 'lon_u'], sustr.swapaxes(1, 2) * 1e3),  # for ROMS
             'svstr': (['sms_time', 'lat_v', 'lon_v'], svstr.swapaxes(1, 2) * 1e3),
             'lon_u': (['lat_u', 'lon_u'], lon_u.transpose()),
             'lat_u': (['lat_u', 'lon_u'], lat_u.transpose()),
             'lon_v': (['lat_v', 'lon_v'], lon_v.transpose()),
             'lat_v': (['lat_v', 'lon_v'], lat_v.transpose())},
            coords={'sms_time': time}
        )
        
        ds_out['sustr'].attrs['long_name'] = "sustr"
        ds_out['sustr'].attrs['units'] = "N m-2"
        ds_out['sustr'].attrs['coordinates'] = "lat_u lon_u"
        ds_out['sustr'].attrs['time'] = "sms_time"
        
        ds_out['svstr'].attrs['long_name'] = "svstr"
        ds_out['svstr'].attrs['units'] = "N m-2"
        ds_out['svstr'].attrs['coordinates'] = "lat_v lon_v"
        ds_out['svstr'].attrs['time'] = "sms_time"
        
        ds_out['sms_time'].attrs['units'] = "days since 0001-01-01 00:00:00"
        
        ds_out.to_netcdf('WindStrMove' + str(params.Wr) + str(int(params.rMax / 1000)) + '.nc')

    return Wx, Wy, Pa
