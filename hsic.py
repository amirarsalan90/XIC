def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma=1):
    if sigma == 0:
        sigma = 0.0001
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)

def HSIC(X, Y):
    n = X.shape[0]
    Xmed = X
    G = torch.sum(Xmed*Xmed, 1).reshape(n,1)
    Q = torch.tile(G, (1,n))
    R = torch.tile(G.T, (n,1))
    
    dists = Q + R - 2*torch.matmul(Xmed, Xmed.T)
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n**2, 1)
    
    s_x = torch.sqrt( 0.5 * torch.median(dists[dists>0]) )
    
    n = Y.shape[0]
    Ymed = Y
    G = torch.sum(Ymed*Ymed, 1).reshape(n,1)
    Q = torch.tile(G, (1,n))
    R = torch.tile(G.T, (n,1))
    
    dists = Q + R - 2*torch.matmul(Ymed, Ymed.T)
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n**2, 1)
    
    s_y = torch.sqrt( 0.5 * torch.median(dists[dists>0]) )
    
    
    m,_ = X.shape #batch size
    K = GaussianKernelMatrix(X,s_x)
    L = GaussianKernelMatrix(Y,s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))
    H = H.float().cuda()
    HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return HSIC
