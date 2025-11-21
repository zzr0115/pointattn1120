import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from utils.train_utils import save_model
from pathlib import Path
import importlib
import munch
import yaml
import os
import random
import numpy as np
import argparse
from dataset import C3D_h5, PCN_pcd
from mylogger import get_logger
from myboard import get_boardwriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR


def epoch_train(net, dataloader, optimizer, device, epoch, exp_name, writer):
    """单个epoch的训练过程"""
    net.train()
    epoch_loss_sum = 0.0
    batch_count = 0
    
    for batch_idx, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        _, inputs, gt = data
        inputs, gt = inputs.float().to(device), gt.float().to(device)
        inputs = inputs.transpose(2, 1).contiguous()
        out2, loss2, net_loss = net(inputs, gt)
        
        if net_loss.dim() > 0:
            net_loss = net_loss.mean()
        
        # 累加用于计算epoch平均loss
        epoch_loss_sum += net_loss.item()
        batch_count += 1
        
        # 计算全局step：epoch * 每个epoch的batch数 + 当前batch索引
        global_step = epoch * len(dataloader) + batch_idx
        
        # 记录到TensorBoard（每个batch）
        writer.add_scalar('train/batch_loss', net_loss.item(), global_step)
        writer.add_scalar('train/fine_loss', loss2.mean().item(), global_step)

        net_loss.backward()
        optimizer.step()
        
        if batch_idx % args.batch_interval_to_print == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'{exp_name} train [epoch {epoch}/{args.end_epoch-1 } ; batch {batch_idx}/{len(dataloader)-1}]')
            logger.info(f'loss_type: {args.loss}; fine_loss: {loss2.mean().item():.6f}; total_loss: {net_loss.item():.6f}; lr: {current_lr:.6f}')
    
    # 记录epoch平均loss
    epoch_avg_loss = epoch_loss_sum / batch_count
    writer.add_scalar('train/epoch_avg_loss', epoch_avg_loss, epoch)


def train(net, dataloader, dataloader_test, optimizer, scheduler, device):
    """训练函数，专注于训练循环逻辑"""
    logger.info(f"Arguments:\n{importlib.import_module('json').dumps(args, indent=2)}")

    # 1. 初始化验证指标和TensorBoard
    metrics = ['cd_p', 'cd_t', 'cd_t_coarse', 'cd_p_coarse']
    best_epoch_losses = {m: (0, np.inf) for m in metrics}
    writer = get_boardwriter(exp_name)

    # 2. 训练循环
    for epoch in range(args.start_epoch, args.end_epoch):
        # 训练一个epoch
        epoch_train(net, dataloader, optimizer, device, epoch, exp_name, writer)
        
        # 每隔1个epoch保存当前模型
        if epoch % args.epoch_interval_to_save == 0:
            logger.info(f"Saving latest epoch trained model...")
            save_model(f'{log_dir}/latest_epoch_network.pth', net)
        
        # 验证
        if epoch % args.epoch_interval_to_val == 0 or epoch == args.end_epoch - 1:
            val(net=net, curr_epoch_num=epoch, metrics=metrics,
                dataloader_test=dataloader_test, best_epoch_losses=best_epoch_losses, 
                writer=writer, device=device)
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('train/learning_rate', current_lr, epoch)
            writer.add_scalar('train/learning_rate_log', np.log10(current_lr), epoch)  # 对数尺度
    
    writer.close()
    logger.info("Training completed!")


def val(net, curr_epoch_num, metrics, dataloader_test, best_epoch_losses, writer, device):
    logger.info(f'Epoch {curr_epoch_num} Testing...')
    net.eval()
    
    # 初始化累加器
    loss_sums = {m: 0.0 for m in metrics}
    batch_count = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader_test):
            label, inputs, gt = data
            inputs = inputs.float().to(device)
            gt = gt.float().to(device)
            inputs = inputs.transpose(2, 1).contiguous()
            result_dict = net(inputs, gt, is_training=False)
            
            # 累加各指标
            for metric in metrics:
                loss_sums[metric] += result_dict[metric].mean().item()
            batch_count += 1
    
        # 计算平均值
        loss_avgs = {m: loss_sums[m] / batch_count for m in metrics}
        
        # 记录验证指标到TensorBoard
        for loss_type, avg_loss in loss_avgs.items():
            writer.add_scalar(f'val/{loss_type}', avg_loss, curr_epoch_num)
        
        # 更新最佳模型
        best_log = []
        for type, (best_epoch, best_loss) in best_epoch_losses.items():
            if loss_avgs[type] < best_loss:
                best_epoch_losses[type] = (curr_epoch_num, loss_avgs[type])
                save_model(f'{log_dir}/best_{type}_network.pth', net)
                logger.info(f'Best {type} network saved!')
                best_log.append(f'best_{type}: {best_epoch_losses[type][1]:.6f} [epoch {best_epoch_losses[type][0]}]')
            else:
                best_log.append(f'best_{type}: {best_loss:.6f} [epoch {best_epoch}]')

        curr_log = [f'curr_{type}: {loss_avgs[type]:.6f}' for type in metrics]

        logger.info('; '.join(curr_log))
        logger.info('; '.join(best_log))   


def result_test():
    # 初始化数据集
    if args.dataset == 'pcn':
        dataset = PCN_pcd(args.pcnpath, prefix="train")
        dataset_test = PCN_pcd(args.pcnpath, prefix="test")
    elif args.dataset == 'c3d':
        dataset = C3D_h5(args.c3dpath, prefix="train")
        dataset_test = C3D_h5(args.c3dpath, prefix="val")
    else:
        raise ValueError('dataset is not exist')
    
    id, inputs, gt = dataset[4002] # 数据集的第4002个数据

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

    # 2. 加载已保存的模型
    model_path = './log/jiaoyanshi/PointAttn_cd_debug_c3d/best_cd_p_network.pth'
    ckpt = torch.load(model_path)
    state_dict = ckpt['net_state_dict']
    
    # 移除可能存在的 'module.' 前缀（多GPU保存的模型）
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    net.eval()

    # 3. 数据预处理
    inputs1 = inputs.float().cuda().unsqueeze(0)
    inputs1 = inputs1.transpose(2, 1).contiguous()  # 将点云数据从 (1, N, 3) 转换为 (1, 3, N)
    gt = gt.float().cuda().unsqueeze(0)

    # 4. 推理
    with torch.no_grad():
        output, loss, net_loss = net(inputs1, gt, is_training=True)

    visualize_pcd = getattr(importlib.import_module("visualize"), "visualize_pcd")
    visualize_pcd(input=inputs, gt=gt.squeeze(0), output=output.squeeze(0)) # output.squeeze(0) 将输出张量从 (1, N, 3) 转换为 (N, 3)
    

if __name__ == "__main__":
    # 1. 解析命令行参数和配置文件
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()

    config_path = Path(arg.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        args = munch.munchify(yaml.safe_load(f))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 2. 设置随机种子（确保可复现性）
    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 3. 设置日志目录
    if args.load_model:
        exp_name = Path(args.load_model).parent.name
        log_dir = Path(args.load_model).parent
    else:
        exp_name = f'{args.model_name}_{args.loss}_{args.flag}_{args.dataset}'
        log_dir = Path(args.work_dir).joinpath(exp_name)
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用 mylogger 获取 logger（自动添加时间戳）
    log_path = log_dir / 'train.log'
    logger = get_logger(name=exp_name, log_file=str(log_path), 
                        level="INFO", add_timestamp=True)
    
    logger.info(f'exp_name: {exp_name}')
    logger.info(f'log save_path: {log_dir}')
    logger.info(f'Random Seed: {seed}')

    # 4. 初始化数据集
    if args.dataset == 'pcn':
        dataset = PCN_pcd(args.pcnpath, prefix="train")
        dataset_test = PCN_pcd(args.pcnpath, prefix="test")
    elif args.dataset == 'c3d':
        dataset = C3D_h5(args.c3dpath, prefix="train")
        dataset_test = C3D_h5(args.c3dpath, prefix="val")
    else:
        raise ValueError('dataset is not exist')
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=int(args.workers),
                            pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, 
                                 shuffle=False, num_workers=int(args.workers),
                                 pin_memory=True)
    
    logger.info(f'Length of train dataset:{len(dataset)}')
    logger.info(f'Length of test dataset:{len(dataset_test)}')

    # 5. 加载模型
    model_module = importlib.import_module(f'.{args.model_name}', 'models')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')
    
    model_config = args.model_configs[args.dataset]
    model = model_module.Model(**model_config)
    
    # 6. 设置device
    # 判断是否使用 DataParallel
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(model)
    else:
        net = model    
    net.to(device)
    
    # 7. 加载已保存的模型权重（如果有的话）
    if args.load_model:
        ckpt = torch.load(args.load_model)
        state_dict = ckpt['net_state_dict']
        
        # 移除可能存在的 'module.' 前缀（多GPU保存的模型）
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        logger.info(f"{args.model_name}'s previous weights loaded.")
    else:
        # 仅在从头训练时初始化权重
        if hasattr(model_module, 'weights_init'):
            model.apply(model_module.weights_init)
            logger.info("Model weights initialized.")

    # 8. 初始化优化器和学习率调度器
    optimizer_class = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer_class(model.parameters(), 
                                    lr=args.lr, 
                                    initial_accumulator_value=args.initial_accum_val)
    else:
        betas = tuple(float(x) for x in args.betas.split(','))
        optimizer = optimizer_class(model.parameters(), 
                                    lr=args.lr, 
                                    weight_decay=args.weight_decay, 
                                    betas=betas)
    
    scheduler = None # 初始化学习率调度器
    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_decay_interval:
            scheduler = StepLR(optimizer, step_size=args.lr_decay_interval, gamma=args.lr_decay_rate)
        elif args.lr_step_decay_epochs:
            milestones = list(int(x) for x in args.lr_step_decay_epochs.split(','))
            gamma = float(args.lr_step_decay_rates.split(',')[0])
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    train(net, dataloader, dataloader_test, optimizer, scheduler, device)
    # result_test()



