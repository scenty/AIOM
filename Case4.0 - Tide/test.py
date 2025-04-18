import numpy as np
from pyproj import Transformer, Proj

class Grid:
    def __init__(self):
        self.lon0 = 111
        self.lon1 = 126
        self.lat0 = 16
        self.lat1 = 33
        self.dx = 5000
        self.dy = 5000
        # 计算UTM带号（适用于北半球）
        self.zone = int((0.5 * (self.lon1 + self.lon0)+ 180) // 6 + 1)

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
        self.xx, self.yy = np.meshgrid(self.xx_, self.yy_)
        self.lon, self.lat = self.utm_to_wgs84(self.xx, self.yy)
        self.Ny, self.Nx = self.lon.shape

grid = Grid()
grid.build_grid()
aa = np.diff(grid.xx)