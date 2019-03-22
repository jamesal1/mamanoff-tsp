import torch
import torch.optim as opt
import matplotlib.pyplot as plt
import time

cuda = True
if cuda:
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

def euclidean_distance(points, diag_one=False):
    ones = torch.diag(Tensor(points.size(0)).fill_(1))
    ssum = (points**2).sum(dim=1)
    diffs = -2 * torch.mm(points, points.t())
    if diag_one:
        res1 = torch.abs(ssum.view(1,-1)+diffs+ssum.view(-1,1)+ones)**.5
    else:
        res1 = torch.abs(ssum.view(1,-1)+diffs+ssum.view(-1,1))**.5
    return res1

def euclidean_c_distance(a,b):

    a2 = (a**2).sum(dim=1)
    b2 = (b**2).sum(dim=1)
    diffs = -2 * torch.mm(a, b.t())
    res1 = torch.abs(a2.view(-1,1)+diffs+b2.view(1,-1))**.5
    return res1



#distance matrix
def mamanoff_distance(points):
    edist = euclidean_distance(points)
    return (edist * points[:,1].view(-1,1) + edist * points[:,1].view(1,-1))/2

def mamanoff_c_distance(a,b):
    edist = euclidean_c_distance(a,b)
    return (edist * a[:,1].view(-1,1) + edist * b[:,1].view(1,-1))/2


def get_grid(size=10):
    ticks = torch.linspace(0,1,steps=(size+1))
    base_points = Tensor([])
    zero_points = Tensor([])
    for y in ticks[1:]:
        for x in ticks:
            base_points = torch.cat([base_points,Tensor([[x,y]])])
    for x in ticks:
        zero_points = torch.cat([zero_points,Tensor([[x,0]])])

    return base_points, zero_points

def get_border(size=10):
    ticks = torch.linspace(0,1,steps=(size+1))
    base_points=Tensor([])
    for y in ticks[1:]:
        for x in [0,1]:
            base_points = torch.cat([base_points,Tensor([[x,y]])])
    for y in [ticks[1],ticks[-1]]:
        for x in ticks[1:-1]:
            base_points = torch.cat([base_points,Tensor([[x,y]])])
    return base_points


def test(size=10,eps=1e-7,iters=2000,pre=2000):
    base_points,zero_points=get_grid(size)
    # print(base_points)
    emb_points = base_points.clone().detach().requires_grad_(True)
    ones = torch.diag(Tensor(emb_points.size(0)).fill_(1))
    ma_dist = mamanoff_distance(base_points) + ones
    zero_ma_dist = mamanoff_c_distance(base_points, zero_points)
    print(zero_ma_dist)
    zero_emb_points=zero_points.clone()
    zero_emb_points[:,0]=.5
    print(zero_emb_points)
    noise = torch.randn_like(emb_points)
    emb_points.data = emb_points + 1e-2 * noise
    # emb_points.data = 1e-1 * noise
    optimizer = opt.SGD([emb_points],lr=1e-3,momentum=.5)
    decay = opt.lr_scheduler.StepLR(optimizer,step_size=1,gamma=1-1e-3)
    optimizer2 = opt.SGD([emb_points],lr=1e-3,momentum=.5)
    decay = opt.lr_scheduler.StepLR(optimizer,step_size=1,gamma=1-1e-3)
    decay2 = opt.lr_scheduler.StepLR(optimizer2,step_size=1,gamma=1-1e-5)
    start = time.time()
    for i in range(iters):

        optimizer.zero_grad()
        optimizer2.zero_grad()
        emb_dist = euclidean_distance(emb_points,True)
        ratio = emb_dist/ma_dist
        # max_vals = torch.max(ratio,dim=1)[0]
        # min_vals = torch.min(ratio,dim=1)[0]

        zero_emb_dist = euclidean_c_distance(emb_points,zero_emb_points)
        ratio_zero = zero_emb_dist/zero_ma_dist
        ratios = torch.cat([ratio,ratio_zero],dim=1)
        max_vals = torch.max(ratios,dim=0)[0]
        min_vals = torch.min(ratios,dim=0)[0]
        max_val = torch.max(max_vals)
        min_val = torch.min(min_vals)
        if i<pre:
            # ll = max_vals/min_val + max_val/min_vals
            ll = max_vals/min_vals
            ll.sum().backward()
            torch.nn.utils.clip_grad_value_([emb_points],1.0)
            optimizer.step()
            decay.step()
        else:
            ll= max_val/min_val
            ll.sum().backward()
            torch.nn.utils.clip_grad_value_([emb_points],1.0)
            optimizer2.step()
            decay2.step()
        if i % 100 == 1:
            print(i)
            print("ratio",max_val,min_val)
            print(max_val/min_val)
            print("point grad",torch.abs(emb_points.grad).sum())
            print("time:",time.time() - start)
            start = time.time()
        # print(ratio)


    emb_points=emb_points.detach().cpu().numpy()
    # print(emb_points)
    plt.scatter(emb_points[:,0],emb_points[:,1],s=10)
    plt.savefig("emb.png")



test()