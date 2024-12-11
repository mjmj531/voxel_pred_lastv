import open3d as o3d
import numpy as np

import open3d as o3d
import numpy as np

def visualize_voxel_grid_to_ply_from_npy(voxel_grid_npy):
    """
    使用 Open3D 可视化从 .npy 文件中加载的体素网格,并将其转换为点云（看上去更好看）。
    
    参数:
        - voxel_grid_npy: numpy 数组，三维体素网格数据，形状为 (D, H, W)。##
        - voxel_size: 每个体素的大小，用于控制可视化的体素大小。
    """
    print(voxel_grid_npy.shape)
    # 找到所有被占据的体素索引（值为1的体素）
    occupied_voxel_indices = np.argwhere(voxel_grid_npy == 1)

    # 等价于：point_cloud = occupied_voxel_indices[:, [0, 1, 2]]
    point_cloud = occupied_voxel_indices[:,]
    # 创建 open3d 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float64))

    # 可视化体素网格
    o3d.visualization.draw_geometries([pcd])


def visualize_voxel_grid_with_color_from_npy(voxel_grid_npy, voxel_size=0.05):
    """
    使用 Open3D 可视化从 .npy 文件中加载的体素网格，并根据深度设置渐变色。
    
    参数:
        - voxel_grid_npy: numpy 数组，三维体素网格数据，形状为 (D, H, W)。
        - voxel_size: 每个体素的大小，用于控制可视化的体素大小。
    """
    # 找到所有被占据的体素索引（值为1的体素）
    occupied_voxel_indices = np.argwhere(voxel_grid_npy == 1)

    # 将体素索引转换为实际坐标
    voxel_coordinates = occupied_voxel_indices * voxel_size

    # 获取 z 轴的最大和最小值
    z_values = voxel_coordinates[:, 2]
    z_min, z_max = z_values.min(), z_values.max()

    # 归一化 z 值到 0-1 区间
    normalized_z = (z_values - z_min) / (z_max - z_min)

    # 根据 z 值创建颜色梯度，颜色从红色到蓝色
    colors = np.zeros((voxel_coordinates.shape[0], 3))
    colors[:, 0] = 1 - normalized_z  # 红色分量：z 值越小越红
    colors[:, 2] = normalized_z      # 蓝色分量：z 值越大越蓝

    # 创建 open3d 点云对象并设置点云的坐标和颜色
    voxel_points = o3d.geometry.PointCloud()
    voxel_points.points = o3d.utility.Vector3dVector(voxel_coordinates)
    voxel_points.colors = o3d.utility.Vector3dVector(colors)

    # 将点云转换为体素网格
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(voxel_points, voxel_size=voxel_size)

    # 可视化体素网格
    o3d.visualization.draw_geometries([voxel_grid], window_name="Voxel Grid with Color Visualization", width=800, height=600)

# 加载保存的体素网格数据
voxel_grid_npy = np.load("dataset_voxel_1109_try/0a761819-05d1-4647-889b-a726747201b1_7_ground_truth.npy")

# 调用可视化函数
visualize_voxel_grid_to_ply_from_npy(voxel_grid_npy)
