"""
将原来的四棱锥形点云改为规则的立方体点云，并保存为 .npy 文件。
参数：
spacing：每个点之间的间距 0.03
ray_points_camera: (N, max_steps, 3) 形状的张量，表示光线的坐标
"""

import numpy as np
import open3d as o3d
import torch

def generate_regular_cube_points_from_ray(ray_points_camera, spacing=0.03):
    """
    基于 ray_points_camera 生成一个规则的三维网格，确保网格覆盖 ray_points_camera 的范围。
    
    :param ray_points_camera: (N, max_steps, 3) 形状的张量，表示光线的坐标
    :param spacing: 每个点之间的间距
    :return: 点云数据，形状为 (M, 3)，其中 M 是网格点的数量
    """
    # 获取 ray_points_camera 的最小值和最大值，确定立方体的尺寸范围
    # Find min/max along each axis (axis 0 is the rays, axis 1 is the points)
    min_coords_x = ray_points_camera[:, :, 0].min()
    min_coords_y = ray_points_camera[:, :, 1].min()
    min_coords_z = ray_points_camera[:, :, 2].min()

    max_coords_x = ray_points_camera[:, :, 0].max()
    max_coords_y = ray_points_camera[:, :, 1].max()
    max_coords_z = ray_points_camera[:, :, 2].max()

    # Create min and max coordinates
    min_coords = torch.tensor([min_coords_x, min_coords_y, min_coords_z])
    max_coords = torch.tensor([max_coords_x, max_coords_y, max_coords_z])

    # 计算立方体的尺寸（宽度、高度、深度）
    dimensions = max_coords - min_coords

    # 计算每个维度的步数
    x_steps = int(np.ceil(dimensions[0] / spacing))
    y_steps = int(np.ceil(dimensions[1] / spacing))
    z_steps = int(np.ceil(dimensions[2] / spacing))
    print(f"x_steps: {x_steps}, y_steps: {y_steps}, z_steps: {z_steps}")

    # 生成规则的三维网格
    x = np.linspace(min_coords[0], max_coords[0], x_steps)
    y = np.linspace(min_coords[1], max_coords[1], y_steps)
    z = np.linspace(min_coords[2], max_coords[2], z_steps)
    
    # 创建网格点
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
    
    # 将网格转换为点云坐标
    points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T
    print(f"points shape: {points.shape}")
    return points

def generate_regular_cube_points_and_extract_roi(ray_points_camera, spacing=0.03):
    """
    基于 ray_points_camera 生成规则点云，同时在生成网格时直接提取深度最小且最靠近中心的 10x10x10 点云。
    
    :param ray_points_camera: (N, max_steps, 3) 形状的张量，表示光线的坐标
    :param spacing: 每个点之间的间距
    :return: 规则立方体点云 (M, 3) 和提取的 ROI 点云 (K, 3)
    """
    # 获取 ray_points_camera 的最小值和最大值，确定立方体的尺寸范围
    min_coords_x = ray_points_camera[:, :, 0].min()
    min_coords_y = ray_points_camera[:, :, 1].min()
    min_coords_z = ray_points_camera[:, :, 2].min()

    max_coords_x = ray_points_camera[:, :, 0].max()
    max_coords_y = ray_points_camera[:, :, 1].max()
    max_coords_z = ray_points_camera[:, :, 2].max()

    # 创建 min 和 max 坐标
    min_coords = torch.tensor([min_coords_x, min_coords_y, min_coords_z])
    max_coords = torch.tensor([max_coords_x, max_coords_y, max_coords_z])

    # 计算立方体的尺寸（宽度、高度、深度）
    dimensions = max_coords - min_coords

    # 计算每个维度的步数
    x_steps = int(np.ceil(dimensions[0] / spacing))
    y_steps = int(np.ceil(dimensions[1] / spacing))
    z_steps = int(np.ceil(dimensions[2] / spacing))
    print(f"x_steps: {x_steps}, y_steps: {y_steps}, z_steps: {z_steps}")

    # 生成规则的三维网格
    x = np.linspace(min_coords[0], max_coords[0], x_steps)
    y = np.linspace(min_coords[1], max_coords[1], y_steps)
    z = np.linspace(min_coords[2], max_coords[2], z_steps)

    # 创建网格点
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
    points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

    print(f"Generated points shape: {points.shape}")

    # 找到深度最小的点所在的平面
    # min_z_index = 0  # 深度最小对应的索引
    # z_selected = z[min_z_index]  # 深度最小的 z 平面

    # 找到 x 和 y 的中心索引
    center_x_index = x_steps // 2
    center_y_index = y_steps // 2

    # 计算 10×10 的范围
    half_size = 10 // 2
    x_start = max(center_x_index - half_size, 0)
    x_end = min(center_x_index + half_size, x_steps)
    y_start = max(center_y_index - half_size, 0)
    y_end = min(center_y_index + half_size, y_steps)

    # 提取 10×10×10 点云
    roi_x = x[x_start:x_end]
    roi_y = y[y_start:y_end]
    roi_z = z[-10:]  # 深度范围的 10 层, 摄像机是由-z轴摄像的，所以这里倒着取点

    # 创建 10×10×10 的网格
    roi_x_grid, roi_y_grid, roi_z_grid = np.meshgrid(roi_x, roi_y, roi_z)
    roi_points = np.vstack([roi_x_grid.ravel(), roi_y_grid.ravel(), roi_z_grid.ravel()]).T

    print(f"Extracted ROI points shape: {roi_points.shape}")
    return points, roi_points


def save_regular_cube_from_ray_to_npy(ray_points_camera, save_whole_points_path, save_roi_points_path, spacing=0.03):
    """
    基于 ray_points_camera 生成规则的立方体点云并保存为 .npy 文件。
    
    :param ray_points_camera: (N, max_steps, 3) 形状的张量，表示光线的坐标
    :param save_path: .npy 文件保存路径
    :param spacing: 每个点之间的间距
    """
    # 生成立方体点云
    points, roi_points = generate_regular_cube_points_and_extract_roi(ray_points_camera, spacing)
    
    # 保存为 .npy 文件
    np.save(save_whole_points_path, points)
    print(f"Point cloud has been saved to {save_whole_points_path}")
    np.save(save_roi_points_path, roi_points)
    print(f"ROI point cloud has been saved to {save_roi_points_path}")

if __name__ == "__main__":
    ray_points_camera = np.load("ray_points_camera_1111.npy")
    print(ray_points_camera.shape)
    save_regular_cube_from_ray_to_npy(ray_points_camera, "knn_test/regular_cube_from_ray_to_npy.npy", "knn_test/roi_cube_from_ray_to_npy.npy", spacing=0.03)