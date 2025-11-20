import torch

class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_model(path, net, net_d=None):
    # 处理 DataParallel/DistributedDataParallel 包装的模型
    net_state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    
    if net_d is not None:
        net_d_state = net_d.module.state_dict() if hasattr(net_d, 'module') else net_d.state_dict()
        torch.save({'net_state_dict': net_state,
                    'D_state_dict': net_d_state}, path)
    else:
        torch.save({'net_state_dict': net_state}, path)








