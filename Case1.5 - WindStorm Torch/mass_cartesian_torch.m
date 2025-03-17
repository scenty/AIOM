def mass_cartesian_torch(H, Z, M, N):
    H0 = H[0]
    Nx_, Ny_ = H0.shape

    # 构造 m_right, m_left
    M0 = M[0]
    m_right = torch.zeros((Nx_, Ny_), dtype=M0.dtype, device=M0.device)
    m_right[:Nx_-1, :] = M0
    m_left = torch.zeros_like(m_right)
    m_left[1:Nx_, :] = M0

    # 构造 n_up, n_down
    N0 = N[0]
    n_up = torch.zeros((Nx_, Ny_), dtype=N0.dtype, device=N0.device)
    n_up[:, :Ny_-1] = N0
    n_down = torch.zeros_like(n_up)
    n_down[:, 1:Ny_] = N0

    CC1_ = dt/dx
    CC2_ = dt/dy

    H1 = H0 - CC1_*(m_right - m_left) - CC2_*(n_up - n_down)

    # 干湿修正（使用 torch.where 保证全为新张量）
    mask_deep = (Z <= -dry_limit)
    H1 = torch.where(mask_deep, torch.zeros_like(H1), H1)

    ZH0 = Z + H0
    ZH1 = Z + H1
    cond = (Z < dry_limit) & (Z > -dry_limit)

    wet_to_dry = cond & (ZH0>0) & ((H1-H0)<0) & (ZH1<=MinWaterDepth)
    c1 = wet_to_dry & (Z>0)
    c2 = wet_to_dry & (Z<=0)
    H1 = torch.where(c1, -Z, H1)
    H1 = torch.where(c2, torch.zeros_like(H1), H1)

    cond_dry = cond & (ZH0<=0)
    c3 = cond_dry & ((H1-H0)>0)
    H1 = torch.where(c3, H1 - H0 - Z, H1)
    c4 = cond_dry & ((H1-H0)<=0) & (Z>0)
    c5 = cond_dry & ((H1-H0)<=0) & (Z<=0)
    H1 = torch.where(c4, -Z, H1)
    H1 = torch.where(c5, torch.zeros_like(H1), H1)

    # 构造新的 H，不再对 H.clone() 进行切片赋值，而是用 stack 构造新张量
    H_new = torch.stack((H0, H1), dim=0)
    return H_new
