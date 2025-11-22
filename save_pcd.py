import open3d as o3d
import importlib
import torch
from dataset import C3D_h5, PCN_pcd

def save_pcd(args):
    """
    加载训练好的模型并可视化预测结果
    
    Args:
        args: 配置参数 (从yaml加载) 包含test_configs字典
    """
    # 从配置中解包测试参数
    dataset_index = args.test_configs['data_index']
    model_path = args.test_configs['model_save_path']
    
    print(f"数据索引: {dataset_index}")
    print(f"模型路径: {model_path}")
    
    # 初始化数据集
    if args.dataset == 'pcn':
        dataset = PCN_pcd(args.pcnpath, prefix="train")
        dataset_test = PCN_pcd(args.pcnpath, prefix="test")
    elif args.dataset == 'c3d':
        dataset = C3D_h5(args.c3dpath, prefix="train")
        dataset_test = C3D_h5(args.c3dpath, prefix="val")
    else:
        raise ValueError('dataset is not exist')
    
    id, inputs, gt = dataset[dataset_index] # 获取指定索引的数据

    model_module = importlib.import_module(f'.{args.model_name}', 'models')
    # 根据 args.dataset 选择对应的模型配置
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
    
    # 移除可能存在的 'module.' 前缀（多GPU保存的模型）
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    net.eval()

    # 数据预处理
    inputs1 = inputs.float().cuda().unsqueeze(0)
    inputs1 = inputs1.transpose(2, 1).contiguous()  # 将点云数据从 (1, N, 3) 转换为 (1, 3, N)
    gt = gt.float().cuda().unsqueeze(0)

    # 推理
    with torch.no_grad():
        output, loss, net_loss = net(inputs1, gt, is_training=True)

    # 保存为PCD文件，供本地查看
    import os
    save_dir = './visualizations'
    os.makedirs(save_dir, exist_ok=True)
    
    # 转换为numpy并创建点云对象
    inputs_np = inputs.cpu().numpy()
    gt_np = gt.squeeze(0).cpu().numpy()
    output_np = output.squeeze(0).cpu().numpy()
    
    # 创建Open3D点云对象（不设置颜色）
    pcd_input = o3d.geometry.PointCloud()
    pcd_input.points = o3d.utility.Vector3dVector(inputs_np)
    
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_np)
    
    pcd_output = o3d.geometry.PointCloud()
    pcd_output.points = o3d.utility.Vector3dVector(output_np)
    
    # 保存PCD文件
    input_path = f'{save_dir}/input_{dataset_index}.pcd'
    gt_path = f'{save_dir}/gt_{dataset_index}.pcd'
    output_path = f'{save_dir}/output_{dataset_index}.pcd'
    
    o3d.io.write_point_cloud(input_path, pcd_input)
    o3d.io.write_point_cloud(gt_path, pcd_gt)
    o3d.io.write_point_cloud(output_path, pcd_output)
    
    print(f"\n点云文件已保存:")
    print(f"  输入: {input_path}")
    print(f"  真值: {gt_path}")
    print(f"  输出: {output_path}")


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
    