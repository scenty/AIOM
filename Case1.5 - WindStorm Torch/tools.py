import torch
import torch.nn.functional as F

def ddx(Y, scheme='edge'):
    #1st order difference along first dimension
    # - Y: 2D tensor
    # - scheme: 'inner' no padding, 'edge' pad so that dimension unchanged
    if scheme=='edge':
        pad = 1
    elif scheme=='inner':
        pad = 0
    
    kernel = torch.tensor([-1,1], dtype=Y.dtype,device=Y.device).unsqueeze(0).unsqueeze(0) 
    
    dMdx = torch.stack([
        F.conv1d(Y[:, i].unsqueeze(0).unsqueeze(0), kernel, padding=pad)
        for i in range(Y.size(1)) ], dim=2).squeeze().t()
        
    return dMdx
    
    
def ddy(Y, scheme='edge'):
    #1st order difference along second dimension
    # - Y: 2D tensor
    # - scheme: 'inner' no padding, 'edge' pad so that dimension unchanged
    if scheme=='edge':
        pad = 1
    elif scheme=='inner':
        pad = 0
    kernel = torch.tensor([-1,1], dtype=Y.dtype,device=Y.device).unsqueeze(0).unsqueeze(0) #[1,1,2]
    dNdy = torch.stack([ 
        F.conv1d(Y[i, :].unsqueeze(0).unsqueeze(0), kernel, padding=pad)
        for i in range(Y.size(0)) ], dim=0).squeeze() #no t()
    return dNdy


def rho2u(Y):
    #from rho-point to u-point (-1 at first dimension)
    # - Y: 2D tensor
    kernel = torch.tensor([.5,.5], dtype=Y.dtype,device=Y.device).unsqueeze(0).unsqueeze(0) #[1,1,2]
    H_u = torch.stack([
        F.conv1d(Y[:, i].unsqueeze(0).unsqueeze(0), kernel, padding=0)
        for i in range(Y.size(1)) ], dim=2).squeeze().t()
    return H_u

def rho2v(Y):
    #from rho-point to v-point (-1 at second dimension)
    # - Y: 2D tensor
    kernel = torch.tensor([.5,.5], dtype=Y.dtype,device=Y.device).unsqueeze(0).unsqueeze(0) #[1,1,2]
    H_v = torch.stack([ 
        F.conv1d(Y[i, :].unsqueeze(0).unsqueeze(0), kernel, padding=0)
        for i in range(Y.size(0)) ], dim=0).squeeze() #no t()
    return H_v

def u2rho(Y):
    #from u-point to rho-point (-1 at first dimension)
    # - Y: 2D tensor
    kernel = torch.tensor([.5,.5], dtype=Y.dtype,device=Y.device).unsqueeze(0).unsqueeze(0) #[1,1,2]
    Yr = torch.stack([
        F.conv1d(Y[:, i].unsqueeze(0).unsqueeze(0), kernel, padding=0)
        for i in range(Y.size(1)) ], dim=2).squeeze().t()
    return Yr

def v2rho(Y):
    #from v-point to rho-point (-1 at second dimension)
    # - Y: 2D tensor
    kernel = torch.tensor([.5,.5], dtype=Y.dtype,device=Y.device).unsqueeze(0).unsqueeze(0) #[1,1,2]
    Yr = torch.stack([ 
        F.conv1d(Y[i, :].unsqueeze(0).unsqueeze(0), kernel, padding=0)
        for i in range(Y.size(0)) ], dim=0).squeeze() #no t()
    return Yr

def ududx_up(M,N,H):
    # Upwind advection
    # only interior points were calculated, M [1:-1]
    #[1 ~ Nx  , 1 ~ Ny  ] for rho
    #[1 ~ Nx-2, 1 ~ Ny-1] for u 
    #[1 ~ Nx-1, 1 ~ Ny-2] for v
    # M = torch.randn([100,51])
    # N = torch.randn([101,50])
    # H = torch.randn([101,51])
    Hu = rho2u(H) #H at u-point 1/2, 3/2, 5/2, 0 ~ Nx, total of Nx
    Mr = u2rho(M) #M at rho-point 1,2,3        1 ~ Nx, total of Nx-1
    
    L12 = torch.sign(M + 1e-12) # 0 ~ Nx-1, total of Nx
    L11 = 1 - L12               # 0 ~ Nx-1, total of Nx
    L13 = -L12 - L11            # 0 ~ Nx-1, total of Nx
    
    t1 = L11[1:-1] * M[2:]**2 /Hu[1:-1]   # Nx - 2
    t2 = L12[1:-1] * M[1:-1]**2/Hu[1:-1]  # 
    t3 = L13[1:-1] * Mr[:-1]**2/H[1:-2]
    
    ududx = t1 + t2 + t3
           
    return ududx

def vdudy_up(M,N,H):
    # Upwind advection
    # only interior points were calculated, M [1:-1]
    #[1 ~ Nx  , 1 ~ Ny  ] for rho
    #[1 ~ Nx-2, 1 ~ Ny-1] for u 
    #[1 ~ Nx-1, 1 ~ Ny-2] for v
    M = torch.randn([100,51])
    N = torch.randn([101,50])
    H = torch.randn([101,51])
    Nu = rho2u(v2rho(N)) #at u-point, [ 0 ~ Nx, 1 ~ Ny-1]
    L12 = torch.sign(Nu + 1e-12) # same with Nu
    L11 = 1 - L12                #
    L13 = -L12 - L11             #
    
    Hu = rho2u(H) #H at u-point 1/2, 3/2, 5/2, 0 ~ Nx, total of Nx
    
    # M all at i + 1/2
    t1 = L11 * M[:, 2:  ] * Nu / Hu[:,1:-1]   # [Nx - 1, Ny - 2]
    t2 = L12 * M[:, 1:-1] * Nu / Hu[:,1:-1]   # 
    t3 = L13 * M[:, :-2]  * Nu / Hu[:,1:-1]
    
    vdudy = t1 + t2 + t3
           
    return vdudy
