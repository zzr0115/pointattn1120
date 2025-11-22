import open3d as o3d
import numpy as np

def visualize_pcd(input_path, gt_path, output_path):
    """
    从PCD文件加载并可视化点云 
    本地使用只需要open3d和numpy
    
    Args:
        input_path: 输入点云PCD文件路径
        gt_path: 真值点云PCD文件路径
        output_path: 输出点云PCD文件路径
    """
    import open3d as o3d
    
    # 加载PCD文件
    pcd_input = o3d.io.read_point_cloud(input_path)
    pcd_gt = o3d.io.read_point_cloud(gt_path)
    pcd_output = o3d.io.read_point_cloud(output_path)
    
    # 设置颜色
    pcd_input.paint_uniform_color([1, 0, 0])    # 红色 - 输入
    pcd_gt.paint_uniform_color([0, 1, 0])       # 绿色 - 真值
    pcd_output.paint_uniform_color([0, 0, 1])   # 蓝色 - 输出
    
    # 可视化
    o3d.visualization.draw_geometries(
        [pcd_input, pcd_gt, pcd_output],
        window_name="Point Cloud: Input(red) GT(green) Output(blue)",
        width=1600, 
        height=900
    )

if __name__ == "__main__":
    from pathlib import Path
    
    # 默认读取 visualizations 目录下的点云文件
    vis_dir = Path("./visualizations")
    
    # 查找目录中的PCD文件
    input_files = list(vis_dir.glob("input_*.pcd"))
    gt_files = list(vis_dir.glob("gt_*.pcd"))
    output_files = list(vis_dir.glob("output_*.pcd"))
    
    # 使用找到的第一组文件
    input_pcd_path = str(input_files[0])
    gt_pcd_path = str(gt_files[0])
    output_pcd_path = str(output_files[0])
    
    print(f"Loading:\n  {input_pcd_path}\n  {gt_pcd_path}\n  {output_pcd_path}")
    
    visualize_pcd(input_pcd_path, gt_pcd_path, output_pcd_path)