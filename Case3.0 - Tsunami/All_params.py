import torch
import xarray as xr

class Params_large:
    def __init__(self, device):
        # Domain parameters
        self.Lx = torch.tensor(800 * 1e3, device=device)
        self.Ly = torch.tensor(400 * 1e3, device=device)
        self.Nx = torch.tensor(100, device=device)
        self.Ny = torch.tensor(50, device=device)
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.depth = torch.tensor(50.0, device=device)

        # Physical constants
        self.g = torch.tensor(9.8, device=device)
        self.rho_water = torch.tensor(1025.0, device=device)
        self.rho_air = torch.tensor(1.2, device=device)
        self.Cd = torch.tensor(2.5e-3, device=device)
        self.manning = torch.tensor(0.15, device=device)
        self.dry_limit = torch.tensor(20.0, device=device)
        self.MinWaterDepth = torch.tensor(0.01, device=device)
        self.FrictionDepthLimit = torch.tensor(5e-3, device=device)
        self.f_cor = torch.tensor(0.0, device=device) #1e-5

        # Time parameters
        self.dt = torch.tensor(100.0, device=device)
        self.NT = torch.tensor(100, device=device)
        self.centerWeighting0 = torch.tensor(0.9998, device=device)

        # Boundary conditions
        self.obc_ele = ['Clo', 'Clo', 'Clo', 'Clo']
        self.obc_u2d = ['Clo', 'Clo', 'Clo', 'Clo'] 
        self.obc_v2d = ['Clo', 'Clo', 'Clo', 'Clo']

        # Temporary variables
        self.CC1 = self.dt / self.dx
        self.CC2 = self.dt / self.dy
        self.CC3 = self.CC1 * self.g
        self.CC4 = self.CC2 * self.g

        # Wind parameters
        self.Wr = torch.tensor(30.0, device=device)
        self.rMax = torch.tensor(50 * 1000, device=device)
        self.typhoon_Vec = torch.tensor([5, 0], device=device)  # m/s
        self.typhoon_Pos = [[200 * 1e3 + itime * self.dt * self.typhoon_Vec[0],
                             200 * 1e3 + itime * self.dt * self.typhoon_Vec[1]]
                            for itime in range(self.NT)]
        self.typhoon_Pos = torch.tensor(self.typhoon_Pos, device=device)


class Params_small:
    def __init__(self, device):
        # Domain parameters
        self.Lx = torch.tensor(800 * 1e3, device=device)
        self.Ly = torch.tensor(400 * 1e3, device=device)
        self.Nx = torch.tensor(20, device=device)
        self.Ny = torch.tensor(10, device=device)
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.depth = torch.tensor(50.0, device=device)

        # Physical constants
        self.g = torch.tensor(9.8, device=device)
        self.rho_water = torch.tensor(1025.0, device=device)
        self.rho_air = torch.tensor(1.2, device=device)
        self.Cd = torch.tensor(2.5e-3, device=device)
        self.manning = torch.tensor(0.15, device=device)
        self.dry_limit = torch.tensor(20.0, device=device)
        self.MinWaterDepth = torch.tensor(0.01, device=device)
        self.FrictionDepthLimit = torch.tensor(5e-3, device=device)
        self.f_cor = torch.tensor(0.0, device=device) #1e-5

        # Time parameters
        self.dt = torch.tensor(100.0, device=device)
        self.NT = torch.tensor(100, device=device)
        self.centerWeighting0 = torch.tensor(0.9998, device=device)

        # Boundary conditions
        self.obc_ele = ['Clo', 'Clo', 'Clo', 'Clo']
        self.obc_u2d = ['Clo', 'Clo', 'Clo', 'Clo']
        self.obc_v2d = ['Clo', 'Clo', 'Clo', 'Clo']


        # Temporary variables
        self.CC1 = self.dt / self.dx
        self.CC2 = self.dt / self.dy
        self.CC3 = self.CC1 * self.g
        self.CC4 = self.CC2 * self.g

        # Wind parameters
        self.Wr = torch.tensor(30.0, device=device)
        self.rMax = torch.tensor(50 * 1000, device=device)
        self.typhoon_Vec = torch.tensor([5, 0], device=device)  # m/s
        self.typhoon_Pos = [[200 * 1e3 + itime * self.dt * self.typhoon_Vec[0],
                             200 * 1e3 + itime * self.dt * self.typhoon_Vec[1]]
                            for itime in range(self.NT)]
        self.typhoon_Pos = torch.tensor(self.typhoon_Pos, device=device)      

class Params_tsunami:
    def __init__(self):
        # ------------------ 新增代码 ------------------
        # 读取 XYZ 数据来确定经纬度网格
        ds_xyz = xr.open_dataset('Data/tsunami_grid_0.1.nc')
        lon = ds_xyz['lon'].values  # 假设单位是度，如果需要转换为物理距离，请根据经纬度转化公式
        lat = ds_xyz['lat'].values
        self.lon = lon 
        self.lat = lat

        # 这里假定 lon、lat 是一维数组，且是规则网格
        self.Nx = len(self.lon) - 1  # 网格点数为维度数-1，如果你希望以点数计算，则直接使用 len(lon)
        self.Ny = len(self.lat) - 1

        # 这里的 Lx, Ly 可以根据经纬度分辨率换算为物理距离（例如：假设1°约111km，或者用更精确的方法）
        # 以下示例仅供参考：
        self.Lx = (lon[-1] - lon[0]) * 111e3  # 单位：米
        self.Ly = (lat[-1] - lat[0]) * 111e3

        # 计算 dx, dy
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        # ------------------------------------------------

        # 其他参数可以保持不变或按需要调整
        self.g = 9.8
        self.rho_water = 1025.0
        self.rho_air = 1.2
        self.Cd = 0
        # 这里仅示例 Manning 系数的设置，后面不再赘述
        self.manning_x = torch.zeros(self.Nx + 1)
        self.manning_x[:] = 0.03

        self.manning = self.manning_x.unsqueeze(1).repeat(1, self.Ny+1)
        self.dry_limit = 100
        self.MinWaterDepth = 0.01
        self.FrictionDepthLimit = 5e-3
        self.f_cor = 0.0

        self.dt = 25
        self.NT = 86400//self.dt
        self.centerWeighting0 = 0.9998

        self.obc_ele = ['Clo', 'Clo', 'Clo', 'Clo']
        self.obc_u2d = ['Clo', 'Clo', 'Clo', 'Clo']
        self.obc_v2d = ['Clo', 'Clo', 'Clo', 'Clo']
        
        self.CC1 = self.dt / self.dx
        self.CC2 = self.dt / self.dy
        self.CC3 = self.CC1 * self.g
        self.CC4 = self.CC2 * self.g

        self.Wr = 30
        self.rMax = 50 * 1000
        self.typhoon_Vec = [5, 0]
        self.typhoon_Pos = [[200*1e3 + itime*self.dt*self.typhoon_Vec[0], 
                             200*1e3 + itime*self.dt*self.typhoon_Vec[1]]
                             for itime in range(self.NT)]