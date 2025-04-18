from pyproj import Transformer, Proj
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#from scipy import ndimage

class Grid:
    def __init__(self, lon0, lat0, lon1, lat1, dx, dy):
        self.lon0 = lon0
        self.lon1 = lon1
        self.lat0 = lat0
        self.lat1 = lat1
        self.dx = dx
        self.dy = dy
        self.rx0_max = 0.15
        self.h_min = 10
        self.h_max = 5000
        # 计算UTM带号（适用于北半球）
        self.zone = int((0.5 * (self.lon1 + self.lon0)+ 180) // 6 + 1)
        self.build_grid()
        self.interp_etopo2()
        self.bound = self.get_bound()

    def wgs84_to_utm(self, lon, lat):
        # 创建UTM投影（椭球体设为WGS84）
        utm_proj = Proj(proj='utm', zone=self.zone, ellps='WGS84', preserve_units=False)
        # 转换坐标（输出单位为米）
        x, y = utm_proj(lon, lat)
        return x, y  # 单位：米

    def utm_to_wgs84(self, x, y, northern=True):
        """UTM坐标转回WGS84经纬度
        Args:
            x/y: UTM坐标（米）
            zone: UTM带号（1-60）
            northern: 是否北半球（南半球设为False）
        """
        proj = Proj(proj='utm', zone=self.zone, ellps='WGS84', south=not northern)
        return proj(x, y, inverse=True)  # 返回经度, 纬度

    def build_grid(self):
        x0, y0 = self.wgs84_to_utm(self.lon0, self.lat0)
        x1, y1 = self.wgs84_to_utm(self.lon1, self.lat1)
        self._Nx = int((x1 - x0) / self.dx)
        self._Ny = int((y1 - y0) / self.dy)
        self.xx_ = np.linspace(x0, x1, self._Nx + 1)
        self.yy_ = np.linspace(y0, y1, self._Ny + 1)
        self.xx, self.yy = np.meshgrid(self.xx_, self.yy_, indexing='ij')
        self.lon, self.lat = self.utm_to_wgs84(self.xx, self.yy)
        self.Nx, self.Ny = self.lon.shape

    def rho2uvp(self, rfield):
        """
        ################################################################
        #
        #   compute the values at u,v and psi points...
        #
        #  Further Information:
        #  http://www.brest.ird.fr/Roms_tools/
        #
        #  This file is part of ROMSTOOLS
        """
        Mp, Lp = rfield.shape
        M = Mp - 1
        L = Lp - 1
        vfield = 0.5 * (rfield[np.arange(0, M), :] + rfield[np.arange(1, Mp), :])
        ufield = 0.5 * (rfield[:, np.arange(0, L)] + rfield[:, np.arange(1, Lp)])
        pfield = 0.5 * (ufield[np.arange(0, M), :] + ufield[np.arange(1, Mp), :])

        return ufield, vfield, pfield

    def dist(self, y0, y1, x0, x1):
        return np.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))

    def get_metrics(self, latu, lonu, latv, lonv):
        """
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #
        #        Compute the pm and pn factors of a grid netcdf file
        #
        #  Further Information:
        #  http://www.brest.ird.fr/Roms_tools/
        #
        #  This file is part of ROMSTOOLS
        """
        Mp, L = latu.shape
        M, Lp = latv.shape
        Lm = L - 1
        Mm = M - 1

        dx = np.zeros([Mp, Lp])
        dy = np.zeros([Mp, Lp])
        dndx = np.zeros([Mp, Lp])
        dmde = np.zeros([Mp, Lp])

        lat1 = latu[:, np.arange(0, Lm)]
        lat2 = latu[:, np.arange(1, L)]
        lon1 = lonu[:, np.arange(0, Lm)]
        lon2 = lonu[:, np.arange(1, L)]

        dx[:, np.arange(1, L)] = self.dist(lat1, lat2, lon1, lon2)

        dx[:, 0] = dx[:, 1]
        dx[:, Lp - 1] = dx[:, L - 1]

        lat1 = latv[np.arange(0, Mm), :]
        lat2 = latv[np.arange(1, M), :]
        lon1 = lonv[np.arange(0, Mm), :]
        lon2 = lonv[np.arange(1, M), :]

        dy[np.arange(1, M), :] = self.dist(lat1, lat2, lon1, lon2)

        dy[0, :] = dy[1, :]
        dy[Mp - 1, :] = dy[M - 1, :]

        pm = 1 / dx
        pn = 1 / dy

        #  dndx and dmde
        # pn2 = pn[1:-2, 2:-1]
        # pn3 = pn[1:-2, 2:-1]
        # dndx[1:-2, 1:-2] = 0.5 * (1 / pn2 - 1 / pn3)
        #
        # pm2 = pm[2:-1, 1:-2]
        # pm3 = pm[2:-1, 1:-2]
        # dmde[1:-2, 1:-2] = 0.5 * (1 / pm2 - 1 / pm3)

        return pm, pn #, dndx, dmde

    def RoughnessMatrix(self, DEP, MSK):
        """
        RoughMat=GRID_RoughnessMatrix(DEP, MSK)

        ---DEP is the bathymetry of the grid
        ---MSK is the mask of the grid
        """

        eta_rho, xi_rho = DEP.shape

        Umat = np.array([[0, 1],
                         [1, 0],
                         [0, -1],
                         [-1, 0]])

        RoughMat = np.zeros(DEP.shape)

        for iEta in range(1, eta_rho - 1):
            for iXi in range(1, xi_rho - 1):
                if (MSK[iEta, iXi] == 1):
                    rough = 0
                    for i in range(4):
                        iEtaB = iEta + Umat[i, 0]
                        iXiB = iXi + Umat[i, 1]
                        if (MSK[iEtaB, iXiB] == 1):
                            dep1 = DEP[iEta, iXi]
                            dep2 = DEP[iEtaB, iXiB]
                            delta = abs((dep1 - dep2) / (dep1 + dep2))
                            rough = np.maximum(rough, delta)

                    RoughMat[iEta, iXi] = rough

        return RoughMat

    def smoothing_Positive_rx0(self, MSK, Hobs, rx0max):
        """
        This program use the direct iterative method from Martinho and Batteen (2006)
        The bathymetry is optimized for a given rx0 factor by increasing it.

        Usage:
        RetBathy = smoothing_Positive_rx0(MSK, Hobs, rx0max)

        ---MSK(eta_rho,xi_rho) is the mask of the grid
             1 for sea
             0 for land
        ---Hobs(eta_rho,xi_rho) is the raw depth of the grid
        ---rx0max is the target rx0 roughness factor
        """

        eta_rho, xi_rho = Hobs.shape
        ListNeigh = np.array([[1, 0],
                              [0, 1],
                              [-1, 0],
                              [0, -1]])
        RetBathy = Hobs.copy()
        nbModif = 0
        tol = 0.000001
        while (True):
            IsFinished = 1
            for iEta in range(eta_rho):
                for iXi in range(xi_rho):
                    if (MSK[iEta, iXi] == 1):
                        for ineigh in range(4):
                            iEtaN = iEta + ListNeigh[ineigh, 0]
                            iXiN = iXi + ListNeigh[ineigh, 1]
                            if (iEtaN <= eta_rho - 1 and iEtaN >= 0 and iXiN <= xi_rho - 1 \
                                    and iXiN >= 0 and MSK[iEtaN, iXiN] == 1):
                                LowerBound = RetBathy[iEtaN, iXiN] * (1 - rx0max) / (1 + rx0max)
                                if ((RetBathy[iEta, iXi] - LowerBound) < -tol):
                                    IsFinished = 0
                                    RetBathy[iEta, iXi] = LowerBound
                                    nbModif = nbModif + 1

            if (IsFinished == 1):
                break

        print('     nbModif=', nbModif)

        return RetBathy

    def get_bound(self):
        Nx, Ny = self.mask.shape[0], self.mask.shape[1]

        mask_e = self.mask[Nx - 1, :]
        lon_e = self.lon[Nx - 1, :]
        lat_e = self.lat[Nx - 1, :]

        mask_w = self.mask[0, :]
        lon_w = self.lon[0, :]
        lat_w = self.lat[0, :]

        mask_n = self.mask[:, Ny - 1]
        lon_n = self.lon[:, Ny - 1]
        lat_n = self.lat[:, Ny - 1]

        mask_s = self.mask[:, 0]
        lon_s = self.lon[:, 0]
        lat_s = self.lat[:, 0]

        return {'east': [mask_e, lon_e, lat_e],
                'west': [mask_w, lon_w, lat_w],
                'north': [mask_n, lon_n, lat_n],
                'south': [mask_s, lon_s, lat_s]}

    def createNiceMap(self, lon, lat, h):

        levels = [-10, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
        # 创建地图
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        # 设置经纬度范围
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
        contour = ax.contourf(
            lon, lat, h,
            levels=levels,
            cmap='terrain',
            extend='both'
        )
        # 添加地理要素
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
        # 添加颜色条
        cbar = plt.colorbar(contour, ax=ax, shrink=0.5)
        cbar.set_label('Elevation (m)')
        # 显示图像
        plt.title('地形可视化')
        plt.show()

    def interp_etopo2(self):
        lon_min = self.lon.min() - 1
        lon_max = self.lon.max() + 1
        lat_min = self.lat.min() - 1
        lat_max = self.lat.max() + 1
        """Get the etopo2 data"""
        etopo2name = r'.\etopo2.nc'
        dataset = nc.Dataset(etopo2name, 'r')
        # 读取经纬度和地形数据
        lon = dataset.variables['lon'][:]
        lat = dataset.variables['lat'][:]
        topo = dataset.variables['topo'][:]
        # 找到经度范围内的索引
        lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        # 找到纬度范围内的索引
        lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        # 提取地形数据
        topo_subset = topo[lat_indices[0]:lat_indices[-1] + 1, lon_indices[0]:lon_indices[-1] + 1]
        # 提取对应的经纬度
        lon_subset = lon[lon_indices]
        lat_subset = lat[lat_indices]
        # 插值
        interp_func = RegularGridInterpolator(
            (lat_subset, lon_subset), topo_subset,
            method='linear', bounds_error=False, fill_value=np.nan
        )
        points = np.column_stack((self.lat.flatten(), self.lon.flatten()))
        self.topo_interp = interp_func(points).reshape(self.lat.shape)
        self.mask = np.ones_like(self.topo_interp, dtype=int)
        self.mask[self.topo_interp > 0] = 0
        #self.mask[201, 0] = 1
        #self.mask[111:116, 154:159] = 1
        self.mask[63:65,0:3]=1
        self.mask[99, 65]=1
        self.topo_interp = - self.topo_interp
        topo_clipped = np.copy(self.topo_interp)
        topo_clipped[topo_clipped < self.h_min] = self.h_min
        topo_clipped[topo_clipped > self.h_max] = self.h_max
        RoughMat0 = self.RoughnessMatrix(topo_clipped, self.mask)
        self.h = self.smoothing_Positive_rx0(self.mask, topo_clipped, self.rx0_max)
        RoughMat1 = self.RoughnessMatrix(self.h, self.mask)
        self.h[self.mask < 0.5] = -25
        # self.h[self.mask > 0.5] = 3840 # flat bottom
        print(f'[原始  ] Max Roughness value: {RoughMat0.max():.4f}')
        print(f'[平滑后] Max Roughness value: {RoughMat1.max():.4f}')
        # self.createNiceMap(self.lon, self.lat, self.h)
        xx_u, xx_v, xx_p = self.rho2uvp(self.xx)
        yy_u, yy_v, yy_p = self.rho2uvp(self.yy)
        self.pm, self.pn = self.get_metrics(yy_u, xx_u, yy_v, xx_v)
        self.f_cor = 4 * np.pi * np.sin(np.pi * self.lat / 180) / (24 * 3600)


        # 原始数据
        # plt.figure(figsize=(12, 5))
        # # plt.subplot(1, 2, 1)
        # # plt.pcolormesh(lon_subset, lat_subset, topo_subset, shading='auto', cmap='terrain')
        # # plt.title('Original Data')
        #
        # # 插值后的数据
        # # plt.subplot(1, 2, 2)
        # plt.pcolormesh(self.lon, self.lat, self.h, shading='auto', cmap='terrain')
        # plt.title('Interpolated Data')
        #
        # plt.colorbar(label='Elevation (m)')
        # plt.tight_layout()
        # plt.show()

#grid = Grid(lon0=111, lon1=126, lat0=16, lat1=33, dx=25000, dy=25000)
# grid.build_grid()
# grid.interp_etopo2()