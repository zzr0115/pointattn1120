import open3d as o3d
import importlib
import torch
from dataset import C3D_h5, PCN_pcd

def save_pcd(args):
    """
    加载训练好的模型并保存所有验证集的预测结果
    
    Args:
        args: 配置参数 (从yaml加载)
    """
    model_path = args.test_configs['model_save_path']
    
    print(f"模型路径: {model_path}")
    
    # 初始化数据集
    if args.dataset == 'pcn':
        dataset_test = PCN_pcd(args.pcnpath, prefix="test")
    elif args.dataset == 'c3d':
        dataset_test = C3D_h5(args.c3dpath, prefix="val")
    else:
        raise ValueError('dataset is not exist')
    
    print(f"验证集样本数: {len(dataset_test)}")

    model_module = importlib.import_module(f'.{args.model_name}', 'models')
    model_config = args.model_configs[args.dataset]
    model = model_module.Model(**model_config)

    # 设置device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(model)
    else:
        net = model
    
    net.to(device)

    # 加载已保存的模型
    ckpt = torch.load(model_path)
    state_dict = ckpt['net_state_dict']
    
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    net.eval()

    # 保存目录
    import os
    save_dir = './visualizations'
    os.makedirs(save_dir, exist_ok=True)
    
    # 遍历所有验证集样本
    print(f"\n开始处理验证集...")
    for idx in range(len(dataset_test)):
        id, inputs, gt = dataset_test[idx]
        
        # 数据预处理，添加 batch 维度，从 (N, 3) 变成 (1, N, 3)
        inputs1 = inputs.float().to(device).unsqueeze(0)
        inputs1 = inputs1.transpose(2, 1).contiguous()
        gt_cuda = gt.float().to(device).unsqueeze(0)

        # 推理
        with torch.no_grad():
            result = net(inputs1, gt_cuda, is_training=False)
            output = result['out2']  # 使用最精细的输出 fine1

        # 转换为numpy
        inputs_np = inputs.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        output_np = output.squeeze(0).detach().cpu().numpy()
        
        # 创建点云对象
        pcd_input = o3d.geometry.PointCloud()
        pcd_input.points = o3d.utility.Vector3dVector(inputs_np)
        
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_np)
        
        pcd_output = o3d.geometry.PointCloud()
        pcd_output.points = o3d.utility.Vector3dVector(output_np)
        
        # 保存PCD文件
        input_path = f'{save_dir}/input_{idx}.pcd'
        gt_path = f'{save_dir}/gt_{idx}.pcd'
        output_path = f'{save_dir}/output_{idx}.pcd'
        
        o3d.io.write_point_cloud(input_path, pcd_input)
        o3d.io.write_point_cloud(gt_path, pcd_gt)
        o3d.io.write_point_cloud(output_path, pcd_output)
        
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(dataset_test)} 个样本")
    
    print(f"\n完成! 所有点云文件已保存到: {save_dir}/")
    print(f"共保存 {len(dataset_test)} 组点云 (input, gt, output)")


if __name__ == "__main__":
    import argparse
    import yaml
    import munch
    from pathlib import Path
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Visualize point cloud completion results')
    parser.add_argument('-c', '--config', default='cfgs/PointAttN.yaml', 
                        help='path to config file')
    arg = parser.parse_args()
    
    # 加载配置文件
    config_path = Path(arg.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        args = munch.munchify(yaml.safe_load(f))

    save_pcd(args)
    