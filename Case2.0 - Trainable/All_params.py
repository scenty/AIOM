import torch

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
        self.NT = torch.tensor(200, device=device)
        self.centerWeighting0 = torch.tensor(0.9998, device=device)

        # Boundary conditions
        self.obc_ele = ['Rad', 'Rad', 'Rad', 'Rad']
        self.obc_u2d = ['Rad', 'Rad', 'Rad', 'Rad'] 
        self.obc_v2d = ['Rad', 'Rad', 'Rad', 'Rad']

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