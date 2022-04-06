import torch

# empirically validate upper noise ceiling bound

def correlate(x, y):
    # 1D correlation
    zx = (x - x.mean()) / x.std(unbiased=True)
    zy = (y - y.mean()) / y.std(unbiased=True)
    return torch.dot(zx, zy) / len(x)


def calc_mean_corr(x, y):
    corr = 0
    for i in range(len(x)):
        corr += correlate(x[i], y)
    corr = corr / len(x)
    return corr


x = torch.rand(2, 30)
y = torch.rand(30)
y.requires_grad = True


def closure():
    opt.zero_grad()
    loss = -calc_mean_corr(x, y)
    loss.backward()
    return loss


opt = torch.optim.LBFGS([y])

for i in range(5):
    opt.step(closure)
    print("optimized solution:", calc_mean_corr(x, y))

print("heuristic solution:", calc_mean_corr(x, x.mean(dim=0)))
