# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 22:12:11 2025

@author: liuy
"""

import os
import argparse
import datetime
import yaml
import numpy as np
import xarray as xr
# - pyTMD
import pyTMD
import timescale.eop
import timescale.time

def compute_pt_tide_ts(pt_coords: tuple,
                       date1: datetime.datetime, date2: datetime.datetime,
                        tide_model_path: str, timestep) -> dict:
    """
    Compute Tide Time Series at the selected Geographic Coordinates
    :param pt_coords: PointCoordinates - (pt_lat, pt_lon) - deg
    :param date1: Initial Time - datetime.datetime
    :param date2: Final Time - datetime.datetime
    :param tide_model: Tidal Model - ['CATS2008', 'AOTIM5', 'FES2014']
    :param tide_model_path: Absolute Path to Tide Model Data
    :param timestep: timestep in seconds.
    :return: Python Dictionary Containing the computed Time Series.
    """
    # - Extract Point Coordinates
    pt_lon, pt_lat = pt_coords
    # - Define Model's Specific Processing Parameters
    grid_file = os.path.join(tide_model_path, 'grid_tpxo8atlas_30_v1')
    model_file_z = os.path.join(tide_model_path, 'hf.tpxo8_atlas_30_v1')
    model_file_uv = os.path.join(tide_model_path, 'uv.tpxo8_atlas_30_v1')
    epsg_code = "WGS84"
    # - pyTDM parameters
    model_format = 'OTIS'
    # - Variable to Read
    var_z = 'z'
    var_uv = ['u', 'v']
    # prepare time var
    # - Other Parameters
    #n_sec_x_hour = 60 * 60
    #n_sec_x_day = 24 * 60 * 60
    t_date_00 = date1
    t_date_11 = date2
    # - Compute difference in time expressed in hours
    nsteps = int((t_date_11 - t_date_00).total_seconds() / timestep) + 1

    # - Calculate Number of days relative to Jan 1, 1992 (48622 MJD)
    # - using datetime
    EPOCH = [1992, 1, 1]
    t_jd_ref = datetime.datetime(year=1992, month=1, day=1, hour=0)
    t_est_tide = [t_date_00 + datetime.timedelta(seconds=timestep*t)
                  for t in range(nsteps)]
    # - Compute Datetime Values for selected date.
    # delta_time = [(t - t_jd_ref).total_seconds() / n_sec_x_day for t in
    #               t_est_tide]
    delta_time = [(t - t_jd_ref).total_seconds() for t in t_est_tide]
    time_in_seconds = [_ - delta_time[0] for _ in delta_time]
    ts = timescale.time.Timescale().from_deltatime(delta_time,
                                                   epoch=EPOCH, standard='utc')
    nt = len(ts)
    nstation = len(pt_lon)
    # -- read tidal constants a
    # nd interpolate to grid points
    amp_z, ph_z, _, c \
        = pyTMD.io.OTIS.extract_constants(pt_lon, pt_lat, grid_file,
                                          model_file_z, epsg_code,
                                          type=var_z,
                                          method='spline',
                                          grid=model_format)
    # -- calculate complex phase in radians for Euler's
    cph_z = -1j * ph_z * np.pi / 180.0
    # -- calculate constituent oscillation
    hc_z = amp_z * np.exp(cph_z)
    TIDE = pyTMD.predict.time_series(ts.tide, hc_z, c,
                                     deltat=0, corrections=model_format)
    # read U-velocity harm
    amp_u, ph_u, _, c \
        = pyTMD.io.OTIS.extract_constants(pt_lon, pt_lat, grid_file,
                                          model_file_uv, epsg_code,
                                          type=var_uv[0],
                                          method='spline',
                                          grid=model_format)
    # -- calculate complex phase in radians for Euler's
    cph_u = -1j * ph_u * np.pi / 180.0
    # -- calculate constituent oscillation
    hc_u = amp_u * np.exp(cph_u)
    # read V-velocity harm
    amp_v, ph_v, _, c \
        = pyTMD.io.OTIS.extract_constants(pt_lon, pt_lat, grid_file,
                                          model_file_uv, epsg_code,
                                          type=var_uv[1],
                                          method='spline',
                                          grid=model_format)
    # -- calculate complex phase in radians for Euler's
    cph_v = -1j * ph_v * np.pi / 180.0
    # -- calculate constituent oscillation
    hc_v = amp_v * np.exp(cph_v)

    # - Compute Tide Time Series
    tidal_z = np.zeros((nstation, nt))
    tidal_u = np.zeros((nstation, nt))
    tidal_v = np.zeros((nstation, nt))
    for ista in range(nstation):
        HC = hc_z[ista, None, :]
        TIDE = pyTMD.predict.time_series(ts.tide, HC, c,
                                         deltat=0, corrections=model_format)
        # calculate values for minor constituents by inferrence
        MINOR = pyTMD.predict.infer_minor(ts.tide, HC, c, deltat=0, corrections=model_format)
        tidal_z[ista, :] = TIDE.data[:] + MINOR.data[:]

        HC_u = hc_u[ista, None, :]
        TIDE_u = pyTMD.predict.time_series(ts.tide, HC_u, c,
                                         deltat=0, corrections=model_format)
        # calculate values for minor constituents by inferrence
        MINOR_u = pyTMD.predict.infer_minor(ts.tide, HC_u, c, deltat=0, corrections=model_format)
        tidal_u[ista, :] = 0.01 * (TIDE_u.data[:] + MINOR_u.data[:]) # m/s --> cm/s

        HC_v = hc_v[ista, None, :]
        TIDE_v = pyTMD.predict.time_series(ts.tide, HC_v, c,
                                         deltat=0, corrections=model_format)
        # calculate values for minor constituents by inferrence
        MINOR_v = pyTMD.predict.infer_minor(ts.tide, HC_v, c, deltat=0, corrections=model_format)
        tidal_v[ista, :] = 0.01 * (TIDE_v.data[:] + MINOR_v.data[:]) # m/s --> cm/s

    return {'tide_z': tidal_z,
            'tide_u': tidal_u,
            'tide_v': tidal_v,
            'tide_time': t_est_tide,
            'time_in_secs': time_in_seconds}

#
# def main() -> None:
#
#     pt_lat = 26
#     pt_lon = 120
#     tide_model = r"TPXO8-atlas"
#     tide_model_path = r"c:\Users\liuy\Desktop\潮位预报\tpxo8_atlas_compact_v1.tar\DATA\tpxo8_atlas"
#     t_date_00 = datetime.datetime(year=2022, month=1, day=1)
#     t_date_11 = datetime.datetime(year=2022, month=1, day=3)
#
#     # - Compute Tide Time Series at the selected location
#     tide_pt = compute_pt_tide_ts((pt_lon, pt_lat), t_date_00, t_date_11, tide_model_path)
#     return tide_pt
#
# if __name__ == '__main__':
#     start_time = datetime.datetime.now()
#     tide_pt = main()
#     end_time = datetime.datetime.now()
#     print(f'# - Computation Time: {end_time - start_time}')