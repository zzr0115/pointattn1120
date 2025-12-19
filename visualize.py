import open3d as o3d
import numpy as np
import importlib

def visualize_pcd(input_path, gt_path, output_path):
    """
    从PCD文件加载并可视化点云 
    本地使用只需要open3d和numpy
    
    Args:
        input_path: 输入点云PCD文件路径
        gt_path: 真值点云PCD文件路径
        output_path: 输出点云PCD文件路径
    """
    # 加载PCD文件
    pcd_input = o3d.io.read_point_cloud(input_path)
    pcd_gt = o3d.io.read_point_cloud(gt_path)
    pcd_output = o3d.io.read_point_cloud(output_path)
    
    # 设置颜色
    pcd_input.paint_uniform_color([1, 0, 0])  # 红色
    pcd_gt.paint_uniform_color([1, 0.5, 0])  # 橘色
    pcd_output.paint_uniform_color([0, 0, 1])  # 蓝色

    # 上下分屏布局
    # 上半部分：inputs和output左右分屏
    pcd_input_shifted = pcd_input.translate((-1, 0.5, 0), relative=False)
    pcd_output_shifted = pcd_output.translate((1, 0.5, 0), relative=False)
    # 下半部分：gt
    pcd_gt_shifted = pcd_gt.translate((0, -0.5, 0), relative=False)
    pcd_list = [pcd_input_shifted, pcd_output_shifted, pcd_gt_shifted]

    vis = o3d.visualization.Visualizer()
    screen_width, screen_height = 1920, 1080
    window_width, window_height = 1600, 900
    left = (screen_width - window_width) // 2
    top = (screen_height - window_height) // 2
    vis.create_window(window_name=
                      "Point Cloud Visualization: Partial(red) vs GT(orange) vs Model_Output(blue)",
                      width=window_width, height=window_height,
                      left=left, top=top)
    vis.get_render_option().background_color = np.array([0.9, 0.9, 0.9])

    for pcd in pcd_list: 
        vis.add_geometry(pcd)

    center = np.array([0.0, 0.0, 0.0])
    ctr = vis.get_view_control()
    time = importlib.import_module("time")
    
    while vis.poll_events(): 
        ctr.set_lookat(center)
        ctr.set_zoom(0.5)
        ctr.rotate(2, 0)
        vis.update_renderer()
        time.sleep(0.02)
    vis.destroy_window()

if __name__ == "__main__":
    from pathlib import Path
    import argparse
    
    # 默认读取 visualizations 目录下的点云文件
    vis_dir = Path("./visualizations")
    
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Visualize point clouds by ID.")
    parser.add_argument("id", type=int, nargs="?", default=0, help="The ID of the sample to visualize (default: 0)")
    args = parser.parse_args()
    
    target_id = args.id

    # 构造文件路径
    input_pcd_path = vis_dir.joinpath(f"input_{target_id}.pcd")
    gt_pcd_path = vis_dir.joinpath(f"gt_{target_id}.pcd")
    output_pcd_path = vis_dir.joinpath(f"output_{target_id}.pcd")
    
    # 检查路径是否存在
    files = [input_pcd_path, gt_pcd_path, output_pcd_path]
    # 缺失的文件路径，如果所有文件都存在，missing 就是一个空列表 []
    missing = [p for p in files if not p.exists()] 
    
    if missing:
        print(f"Error: Files not found: {[str(p) for p in missing]}")
    else:
        print(f"Loading ID {target_id}:\n  {input_pcd_path}\n  {gt_pcd_path}\n  {output_pcd_path}")
        visualize_pcd(str(input_pcd_path), str(gt_pcd_path), str(output_pcd_path))