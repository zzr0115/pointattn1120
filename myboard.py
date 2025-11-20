from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

def get_boardwriter(script_name=None):
    """创建 TensorBoard SummaryWriter
    Args:
        script_name: 脚本名称，如果为 None 则使用调用者的文件名
    Returns:
        SummaryWriter 实例
    """
    timestamp = datetime.now().strftime("%y-%m-%d_%H%M%S")
    if script_name is None:
        # 使用调用者的文件名
        import inspect
        frame = inspect.currentframe().f_back
        script_name = Path(frame.f_code.co_filename).stem
    log_dir = f"runs/{script_name}/ep_{timestamp}"
    return SummaryWriter(log_dir=log_dir)

if __name__ == "__main__":
    print("Hello from example-torch!")
    # 使用 tensorboard 记录训练过程
    # with get_boardwriter("main") as writer:
    #     for step in range(10):
    #         writer.add_scalar(tag="metrics/accuracy", 
    #                           scalar_value=0.5 + step, global_step=step)
    #         writer.add_scalar(tag="metrics/loss", 
    #                           scalar_value=1 + step, global_step=step)
    
    