import torch

def save_model(path, net, net_d=None):
    # 处理 DataParallel/DistributedDataParallel 包装的模型
    net_state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    
    if net_d is not None:
        net_d_state = net_d.module.state_dict() if hasattr(net_d, 'module') else net_d.state_dict()
        torch.save({'net_state_dict': net_state,
                    'D_state_dict': net_d_state}, path)
    else:
        torch.save({'net_state_dict': net_state}, path)
