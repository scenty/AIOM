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

class Params_tsunami:
    def __init__(self, device):
        # 读取经纬度信息（假设来自一个 netCDF 文件）
        ds_xyz = xr.open_dataset('Data/tsunami_grid_1.0.nc')
        lon = ds_xyz['lon'].values  # 单位：度
        lat = ds_xyz['lat'].values
        self.lon = lon
        self.lat = lat
        
        # 这里假定 lon, lat 是一维数组，构造网格时点数为 len(lon) 和 len(lat)
        # 如果希望用格点数-1作为网格单元数，则转换为 int 类型
        self.Nx = int(len(lon) - 1)
        self.Ny = int(len(lat) - 1)
        
        # 将经纬度差换算为物理距离（假设 1° 约 111 km）
        self.Lx = (lon[-1] - lon[0]) * 111e3   # 单位：米
        self.Ly = (lat[-1] - lat[0]) * 111e3     # 单位：米
        
        # 计算网格间距（这里保留为浮点数）
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        
        # 物理常数，转换为 tensor 并放到指定设备上（采用 dtype=torch.float64）
        self.g = torch.tensor(9.8, device=device, dtype=torch.float64)
        self.rho_water = torch.tensor(1025.0, device=device, dtype=torch.float64)
        self.rho_air = torch.tensor(1.2, device=device, dtype=torch.float64)
        self.Cd = torch.tensor(0.0, device=device, dtype=torch.float64)  # 可按需要修改为2.5e-3
        
        # Manning 系数：构造一个 Nx+1 x Ny+1 的矩阵，并放到 device 上
        manning_x = torch.full((self.Nx+1,), 0.01, dtype=torch.float64, device=device)
        self.manning = manning_x.unsqueeze(1).repeat(1, self.Ny+1)
        
        # 其它物理、数值参数
        self.dry_limit = 0.2
        self.MinWaterDepth = torch.tensor(0.01, device=device, dtype=torch.float64)
        self.FrictionDepthLimit = torch.tensor(5e-3, device=device, dtype=torch.float64)
        self.f_cor = torch.tensor(0.0, device=device, dtype=torch.float64)
        
        # 时间参数
        self.dt = torch.tensor(25.0, device=device, dtype=torch.float64)
        self.NT = 1200 #int(40500 // 60)-2  # 或者直接存为 python 整数
        self.centerWeighting0 = torch.tensor(0.9998, device=device, dtype=torch.float64)
        
        # 边界条件设置（保持列表形式即可）
        self.obc_ele = ['Clo', 'Clo', 'Clo', 'Clo']
        self.obc_u2d = ['Clo', 'Clo', 'Clo', 'Clo']
        self.obc_v2d = ['Clo', 'Clo', 'Clo', 'Clo']
        
        # 计算一些辅助系数（注意 dt, dx, dy 为浮点数或 tensor，此处直接做除法即可）
        self.CC1 = self.dt / self.dx
        self.CC2 = self.dt / self.dy
        self.CC3 = self.CC1 * self.g
        self.CC4 = self.CC2 * self.g
        
        # 如果需要设置台风参数（此部分可根据需要修改）
        self.Wr = torch.tensor(30.0, device=device, dtype=torch.float64)
        self.rMax = torch.tensor(50 * 1000, device=device, dtype=torch.float64)
        self.typhoon_Vec = torch.tensor([5.0, 0.0], device=device, dtype=torch.float64)  # m/s
        # 构造每个时间步的台风位置：此处生成 NT x 2 的张量
        typhoon_positions = []
        for itime in range(self.NT):
            pos_x = 200e3 + itime * self.dt.item() * self.typhoon_Vec[0].item()
            pos_y = 200e3 + itime * self.dt.item() * self.typhoon_Vec[1].item()
            typhoon_positions.append([pos_x, pos_y])
        self.typhoon_Pos = torch.tensor(typhoon_positions, device=device, dtype=torch.float64)
# class Params_small:
#     def __init__(self, device):
#         # Domain parameters
#         self.Lx = torch.tensor(800 * 1e3, device=device)
#         self.Ly = torch.tensor(400 * 1e3, device=device)
#         self.Nx = torch.tensor(20, device=device)
#         self.Ny = torch.tensor(10, device=device)
#         self.dx = self.Lx / self.Nx
#         self.dy = self.Ly / self.Ny
#         self.depth = torch.tensor(50.0, device=device)

#         # Physical constants
#         self.g = torch.tensor(9.8, device=device)
#         self.rho_water = torch.tensor(1025.0, device=device)
#         self.rho_air = torch.tensor(1.2, device=device)
#         self.Cd = torch.tensor(2.5e-3, device=device)
#         self.manning = torch.tensor(0.15, device=device)
#         self.dry_limit = torch.tensor(20.0, device=device)
#         self.MinWaterDepth = torch.tensor(0.01, device=device)
#         self.FrictionDepthLimit = torch.tensor(5e-3, device=device)
#         self.f_cor = torch.tensor(0.0, device=device) #1e-5

#         # Time parameters
#         self.dt = torch.tensor(100.0, device=device)
#         self.NT = torch.tensor(100, device=device)
#         self.centerWeighting0 = torch.tensor(0.9998, device=device)

#         # Boundary conditions
#         self.obc_ele = ['Clo', 'Clo', 'Clo', 'Clo']
#         self.obc_u2d = ['Clo', 'Clo', 'Clo', 'Clo']
#         self.obc_v2d = ['Clo', 'Clo', 'Clo', 'Clo']


#         # Temporary variables
#         self.CC1 = self.dt / self.dx
#         self.CC2 = self.dt / self.dy
#         self.CC3 = self.CC1 * self.g
#         self.CC4 = self.CC2 * self.g

#         # Wind parameters
#         self.Wr = torch.tensor(30.0, device=device)
#         self.rMax = torch.tensor(50 * 1000, device=device)
#         self.typhoon_Vec = torch.tensor([5, 0], device=device)  # m/s
#         self.typhoon_Pos = [[200 * 1e3 + itime * self.dt * self.typhoon_Vec[0],
#                              200 * 1e3 + itime * self.dt * self.typhoon_Vec[1]]
#                             for itime in range(self.NT)]
#         self.typhoon_Pos = torch.tensor(self.typhoon_Pos, device=device)      


# class Params_tsunami:
#     def __init__(self, device):
#         # 读取经纬度信息（假设来自一个 netCDF 文件）
#         ds_xyz = xr.open_dataset('Data/tsunami_grid_1.nc')
#         lon = ds_xyz['lon'].values  # 单位：度
#         lat = ds_xyz['lat'].values
#         self.lon = lon
#         self.lat = lat
        
#         # 这里假定 lon, lat 是一维数组，构造网格时点数为 len(lon) 和 len(lat)
#         # 如果希望用格点数-1作为网格单元数，则转换为 int 类型
#         self.Nx = int(len(lon) - 1)
#         self.Ny = int(len(lat) - 1)
        
#         # 将经纬度差换算为物理距离（假设 1° 约 111 km）
#         self.Lx = (lon[-1] - lon[0]) * 111e3   # 单位：米
#         self.Ly = (lat[-1] - lat[0]) * 111e3     # 单位：米
        
#         # 计算网格间距（这里保留为浮点数）
#         self.dx = self.Lx / self.Nx
#         self.dy = self.Ly / self.Ny
        
#         # 物理常数，转换为 tensor 并放到指定设备上（采用 dtype=torch.float64）
#         self.g = torch.tensor(9.8, device=device, dtype=torch.float64)
#         self.rho_water = torch.tensor(1025.0, device=device, dtype=torch.float64)
#         self.rho_air = torch.tensor(1.2, device=device, dtype=torch.float64)
#         self.Cd = torch.tensor(0.0, device=device, dtype=torch.float64)  # 可按需要修改为2.5e-3
        
#         # Manning 系数：构造一个 Nx+1 x Ny+1 的矩阵，并放到 device 上
#         manning_x = torch.full((self.Nx+1,), 0.03, dtype=torch.float64, device=device)
#         self.manning = manning_x.unsqueeze(1).repeat(1, self.Ny+1)
        
#         # 其它物理、数值参数
#         self.dry_limit = torch.tensor(100.0, device=device, dtype=torch.float64)
#         self.MinWaterDepth = torch.tensor(0.01, device=device, dtype=torch.float64)
#         self.FrictionDepthLimit = torch.tensor(5e-3, device=device, dtype=torch.float64)
#         self.f_cor = torch.tensor(0.0, device=device, dtype=torch.float64)
        
#         # 时间参数
#         self.dt = torch.tensor(60.0, device=device, dtype=torch.float64)
#         self.NT = int(40500 // 60)  # 或者直接存为 python 整数
#         self.centerWeighting0 = torch.tensor(0.9998, device=device, dtype=torch.float64)
        
#         # 边界条件设置（保持列表形式即可）
#         self.obc_ele = ['Clo', 'Clo', 'Clo', 'Clo']
#         self.obc_u2d = ['Clo', 'Clo', 'Clo', 'Clo']
#         self.obc_v2d = ['Clo', 'Clo', 'Clo', 'Clo']
        
#         # 计算一些辅助系数（注意 dt, dx, dy 为浮点数或 tensor，此处直接做除法即可）
#         self.CC1 = self.dt / self.dx
#         self.CC2 = self.dt / self.dy
#         self.CC3 = self.CC1 * self.g
#         self.CC4 = self.CC2 * self.g
        
#         # 如果需要设置台风参数（此部分可根据需要修改）
#         self.Wr = torch.tensor(30.0, device=device, dtype=torch.float64)
#         self.rMax = torch.tensor(50 * 1000, device=device, dtype=torch.float64)
#         self.typhoon_Vec = torch.tensor([5.0, 0.0], device=device, dtype=torch.float64)  # m/s
#         # 构造每个时间步的台风位置：此处生成 NT x 2 的张量
#         typhoon_positions = []
#         for itime in range(self.NT):
#             pos_x = 200e3 + itime * self.dt.item() * self.typhoon_Vec[0].item()
#             pos_y = 200e3 + itime * self.dt.item() * self.typhoon_Vec[1].item()
#             typhoon_positions.append([pos_x, pos_y])
#         self.typhoon_Pos = torch.tensor(typhoon_positions, device=device, dtype=torch.float64)

# class Params_tsunami:
#     def __init__(self, device):
#         # 读取经纬度信息（假设来自一个 netCDF 文件）
#         ds_xyz = xr.open_dataset('Data/tsunami_grid_1.nc')
#         lon = ds_xyz['lon'].values  # 单位：度
#         lat = ds_xyz['lat'].values
#         self.lon = lon
#         self.lat = lat
        
#         # 这里假定 lon, lat 是一维数组，构造网格时点数为 len(lon) 和 len(lat)
#         # 如果希望用格点数-1作为网格单元数，则转换为 int 类型
#         self.Nx = int(len(lon) - 1)
#         self.Ny = int(len(lat) - 1)
        
#         # 将经纬度差换算为物理距离（假设 1° 约 111 km）
#         self.Lx = (lon[-1] - lon[0]) * 111e3   # 单位：米
#         self.Ly = (lat[-1] - lat[0]) * 111e3     # 单位：米
        
#         # 计算网格间距（这里保留为浮点数）
#         self.dx = self.Lx / self.Nx
#         self.dy = self.Ly / self.Ny
        
#         # 物理常数，转换为 tensor 并放到指定设备上（采用 dtype=torch.float64）
#         self.g = torch.tensor(9.8, device=device, dtype=torch.float64)
#         self.rho_water = torch.tensor(1025.0, device=device, dtype=torch.float64)
#         self.rho_air = torch.tensor(1.2, device=device, dtype=torch.float64)
#         self.Cd = torch.tensor(0.0, device=device, dtype=torch.float64)  # 可按需要修改为2.5e-3
        
#         # Manning 系数：构造一个 Nx+1 x Ny+1 的矩阵，并放到 device 上
#         manning_x = torch.full((self.Nx+1,), 0.03, dtype=torch.float64, device=device)
#         self.manning = manning_x.unsqueeze(1).repeat(1, self.Ny+1)
        
#         # 其它物理、数值参数
#         self.dry_limit = 0.2
#         self.MinWaterDepth = torch.tensor(0.01, device=device, dtype=torch.float64)
#         self.FrictionDepthLimit = torch.tensor(5e-3, device=device, dtype=torch.float64)
#         self.f_cor = torch.tensor(0.0, device=device, dtype=torch.float64)
        
#         # 时间参数
#         self.dt = torch.tensor(60.0, device=device, dtype=torch.float64)
#         self.NT = int(40500 // 60)  # 或者直接存为 python 整数
#         self.centerWeighting0 = torch.tensor(0.9998, device=device, dtype=torch.float64)
        
#         # 边界条件设置（保持列表形式即可）
#         self.obc_ele = ['Clo', 'Clo', 'Clo', 'Clo']
#         self.obc_u2d = ['Clo', 'Clo', 'Clo', 'Clo']
#         self.obc_v2d = ['Clo', 'Clo', 'Clo', 'Clo']
        
#         # 计算一些辅助系数（注意 dt, dx, dy 为浮点数或 tensor，此处直接做除法即可）
#         self.CC1 = self.dt / self.dx
#         self.CC2 = self.dt / self.dy
#         self.CC3 = self.CC1 * self.g
#         self.CC4 = self.CC2 * self.g
        
#         # 如果需要设置台风参数（此部分可根据需要修改）
#         self.Wr = torch.tensor(30.0, device=device, dtype=torch.float64)
#         self.rMax = torch.tensor(50 * 1000, device=device, dtype=torch.float64)
#         self.typhoon_Vec = torch.tensor([5.0, 0.0], device=device, dtype=torch.float64)  # m/s
#         # 构造每个时间步的台风位置：此处生成 NT x 2 的张量
#         typhoon_positions = []
#         for itime in range(self.NT):
#             pos_x = 200e3 + itime * self.dt.item() * self.typhoon_Vec[0].item()
#             pos_y = 200e3 + itime * self.dt.item() * self.typhoon_Vec[1].item()
#             typhoon_positions.append([pos_x, pos_y])
#         self.typhoon_Pos = torch.tensor(typhoon_positions, device=device, dtype=torch.float64)