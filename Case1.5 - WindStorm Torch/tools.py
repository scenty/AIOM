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
    
def ududx_up(M,N,H):
    #1st order difference along first dimension
    # - Y: 2D tensor
    # - scheme: 'inner' no padding, 'edge' pad so that dimension unchanged
    Hu = rho2u(H) #H at u-point 1/2, 3/2, 5/2
    Mr = u2rho(M) #M at rho-point 1,2,3
    
    L12 = torch.sign(M + 1e-12)
    L11 = 1 - L12
    L13 = -L12 - L11
    
    ududx = L11 * M[1:]**2/H[1:] + L12 * Mr[:-1]**2/Hu[:-1] + L13 * M**2/Hu[1:]
           
    return ududx
    
def vdudy_up(M,N,H):
    #1st order difference along first dimension
    # - Y: 2D tensor
    # - scheme: 'inner' no padding, 'edge' pad so that dimension unchanged
    Hu = rho2u(H) #H at u-point 1/2, 3/2, 5/2
    Mr = u2rho(M) #M at rho-point 1,2,3
    
    L12 = torch.sign(M + 1e-12)
    L11 = 1 - L12
    L13 = -L12 - L11
    
    ududx = L11 * M[1:]**2/H[1:] + L12 * Mr[:-1]**2/Hu[:-1] + L13 * M**2/Hu[1:]
           
    return ududx