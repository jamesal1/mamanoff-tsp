import torch
import torch.optim as opt
import matplotlib.pyplot as plt
import time


def euclidean_distance(points, diag_one=False):
    ones = torch.diag(torch.cuda.FloatTensor(points.size(0)).fill_(1))
    # ones = torch.diag(torch.ones(points.size(0)))
    ssum = (points**2).sum(dim=1)
    diffs = -2 * torch.mm(points, points.t())
    if diag_one:
        res1 = torch.abs(ssum.view(1,-1)+diffs+ssum.view(-1,1)+ones)**.5
    else:
        res1 = torch.abs(ssum.view(1,-1)+diffs+ssum.view(-1,1))**.5
    return res1
    ssum = ssum.view(1,-1).repeat((ssum.size(0),1))
    res2 = (ssum+diffs+ssum.t())**.5



#distance matrix
def mamanoff_distance(points):
    edist = euclidean_distance(points)
    return (edist * points[:,1].view(-1,1) + edist * points[:,1].view(1,-1))/2



def test(size=10,eps=1e-7):
    ticks = torch.linspace(0,1,steps=(size+1))
    base_points=torch.cuda.FloatTensor([])
    # base_points=torch.Tensor([])
    for y in ticks[1:]:
        for x in ticks:
            base_points = torch.cat([base_points,torch.cuda.FloatTensor([[x,y]])])
            # base_points = torch.cat([base_points,torch.Tensor([[x,y]])])
    emb_points = base_points.clone().detach().requires_grad_(True)
    ones = torch.diag(torch.ones(emb_points.size(0))).cuda()
    # ones = torch.diag(torch.ones(emb_points.size(0)))
    ma_dist = mamanoff_distance(base_points) + ones

    # noise = torch.randn_like(emb_points)
    # emb_points.data = emb_points + 1e-2 * noise
    # emb_points.data = 1e-1 * noise

    optimizer = opt.SGD([emb_points],lr=1e-4,momentum=.9)
    start = time.time()
    for i in range(10000):

        optimizer.zero_grad()
        emb_dist = euclidean_distance(emb_points,True)
        ratio = emb_dist/ma_dist
        ll = torch.max(ratio)/torch.min(ratio)
        ll.backward()
        if i % 100 == 1:
            print(torch.max(ratio),torch.min(ratio))
            print("point grad",torch.abs(emb_points.grad).sum())
            print("time:",time.time() - start)
            start = time.time()
        # print(ratio)
        torch.nn.utils.clip_grad_value_([emb_points],1.0)
        optimizer.step()
    emb_points=emb_points.detach().cpu().numpy()
    print(emb_points)
    plt.scatter(emb_points[:,0],emb_points[:,1])
    plt.savefig("emb.png")



test()