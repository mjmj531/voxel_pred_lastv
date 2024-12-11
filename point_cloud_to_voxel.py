# 这种方法可能将点云中的多个点映射到同一个格子中，导致变为标准网格之后的点只会少不会多
# 1129 先用ray_points_camera计算出体素网格的大小
# 再把不规则点云转换为体素网格voxel
# 最后再把体素网格转换为规则点云
# 可能出现精度问题和偏差(voxel是一堆格子)

# 输入的不规则点云是resnet预测的点云

import numpy as np
import os
import torch

def calculate_voxel_grid_size(ray_points_camera, spacing = 0.03):
    """
    计算ray_points_camera对应的体素网格的大小。
    """
    # 获取 ray_points_camera 的最小值和最大值，确定立方体的尺寸范围
    min_coords_x = ray_points_camera[:, :, 0].min()
    min_coords_y = ray_points_camera[:, :, 1].min()
    min_coords_z = ray_points_camera[:, :, 2].min()
    max_coords_x = ray_points_camera[:, :, 0].max()
    max_coords_y = ray_points_camera[:, :, 1].max()
    max_coords_z = ray_points_camera[:, :, 2].max()

    # 创建 min 和 max 坐标
    min_coords = np.array([min_coords_x, min_coords_y, min_coords_z])
    max_coords = np.array([max_coords_x, max_coords_y, max_coords_z])

    # 计算立方体的尺寸（宽度、高度、深度）
    dimensions = max_coords - min_coords

    # 计算每个维度的步数
    x_steps = int(np.ceil(dimensions[0] / spacing))
    y_steps = int(np.ceil(dimensions[1] / spacing))
    z_steps = int(np.ceil(dimensions[2] / spacing))
    print(f"x_steps: {x_steps}, y_steps: {y_steps}, z_steps: {z_steps}")

    # 体素网格的尺寸和体素大小
    grid_size = (x_steps, y_steps, z_steps) 
    return grid_size, min_coords

def ply_to_voxel_grid(points, grid_size, min_bound, spacing=0.03):
    voxel_size_x = voxel_size_y = voxel_size_z = spacing
    # 将点云坐标转换为体素网格的坐标
    voxel_coords = ((points - min_bound) / np.array([voxel_size_x, voxel_size_y, voxel_size_z])).astype(int)
    # 确保坐标不越界
    voxel_coords = np.clip(voxel_coords, [0, 0, 0], [grid_size[0] - 1, grid_size[1] - 1, grid_size[2] - 1])
    # 创建空的体素网格
    voxel_grid = np.zeros(grid_size, dtype=np.uint8)

    # 使用 NumPy 的高级索引设置体素值为 1
    voxel_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1
    return voxel_grid

def voxel_grid_to_regular_ply(voxel_grid, min_bound, save_folder, identifier, last_part, spacing=0.03):
    """
    将体素网格转换为规则点云。
    """
    # 获取体素网格的尺寸
    # x_steps, y_steps, z_steps = voxel_grid.shape
    # 获取体素网格中所有占据体素的索引
    occupied_voxels = np.argwhere(voxel_grid == 1)
    # 将体素坐标转换为实际的空间坐标
    pointcloud = occupied_voxels * spacing + min_bound
    # 保存规则点云
    save_path = os.path.join(save_folder, f"{identifier}_{last_part}_regular_pointcloud.npy")
    np.save(save_path, pointcloud)
    print(f"规则点云已保存到 {save_path}")
    return pointcloud

if __name__ == '__main__':
    ray_points_camera = np.load("ray_points_camera_1111.npy")
    # 点云数据路径
    pointcloud_floder = "pred_outcome_for_npy_and_ply/predicted_N_3_npy_pw_0.64_origin_lr_0.0008_bs_16_epoch_28"

    pointcloud_path = "a5af701c-1825-40ce-a9d7-0b7ed08fb43b_4_predicted_N_3.npy"
    pointcloud = os.path.join(pointcloud_floder, pointcloud_path)

    base_name = os.path.splitext(os.path.basename(pointcloud_path))[0]
            
    identifier = base_name.split('_')[0]
    print(f"identifier is {identifier}")
    last_part = base_name.split('_')[1]
    print(f"Last part is {last_part}")

    # 体素网格大小
    spacing = 0.03
    grid_size, min_bound = calculate_voxel_grid_size(ray_points_camera, spacing)
    # 加载点云数据
    points = np.load(pointcloud)
    # 将点云转换为体素网格
    voxel_grid = ply_to_voxel_grid(points, grid_size, min_bound, spacing)
    # 将体素网格转换为规则点云并保存
    save_folder = "knn_test"
    os.makedirs(save_folder, exist_ok=True)
    voxel_grid_to_regular_ply(voxel_grid, min_bound, save_folder, identifier, last_part, spacing)


# def pointcloud_to_voxel_grid(pointcloud, voxel_size, global_min_bound, grid_shape):
#     """
#     参数:
#         - pointcloud: numpy array, 点云数据，形状为 (N, 3)。
#         - voxel_size: float, 体素的边长。
#         - global_min_bound: numpy array, 全局最小边界，用于统一体素位置。
#         - grid_shape: tuple, 体素网格的形状 (D, H, W)。

#     返回:
#         - voxel_grid: numpy array, 体素网格。
#     """
#     voxel_grid = np.zeros(grid_shape, dtype=np.float16)

#     # 将每个点映射到全局体素网格
#     voxel_indices = ((pointcloud - global_min_bound) / voxel_size).astype(int)

#     # 过滤掉超出体素网格的点
#     valid_indices = (
#         (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < grid_shape[0]) &
#         (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < grid_shape[1]) &
#         (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < grid_shape[2])
#     )
#     voxel_indices = voxel_indices[valid_indices]

#     # 将有效的体素索引位置设置为1
#     voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

#     return voxel_grid


# # 遍历所有点云文件，进行体素化处理
# for filename in os.listdir(input_folder):
#     if filename.endswith(".npy"):
#         # 加载点云数据
#         pointcloud = np.load(os.path.join(input_folder, filename))

#         # 将点云转换为全局体素网格
#         voxel_grid = pointcloud_to_voxel_grid(pointcloud, voxel_size, global_min_bound, grid_shape)

#         # 保存体素网格
#         output_path = os.path.join(output_folder, filename)
#         np.save(output_path, voxel_grid)
        
#         print(f"已处理并保存：{filename}")

# print("所有文件已处理完成！")

# def pointcloud_to_voxel_grid(pointcloud, voxel_size=0.05):
#     """
#     将点云数据转换为体素网格。

#     参数:
#         - pointcloud: numpy array, 点云数据，形状为 (N, 3)。
#         - voxel_size: float, 体素的边长。

#     返回:
#         - voxel_grid: numpy array, 体素网格，3D 数组。
#     """
#     # 计算点云的边界
#     min_bound = pointcloud.min(axis=0)
#     max_bound = pointcloud.max(axis=0)

#     # 计算网格的大小
#     grid_size = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
#     voxel_grid = np.zeros(grid_size, dtype=np.uint8)

#     # 将每个点映射到体素网格上
#     voxel_indices = ((pointcloud - min_bound) / voxel_size).astype(int)

#     # 使用 np.clip 将索引限制在 voxel_grid 的边界范围内
#     voxel_indices = np.clip(voxel_indices, 0, grid_size - 1)
#     voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

#     return voxel_grid

# # 示例用法
# voxel_grid = pointcloud_to_voxel_grid(np.load("dataset_point_cloud_1109/0a761819-05d1-4647-889b-a726747201b1_2_input_info.npy"))
# np.save("0a761819-05d1-4647-889b-a726747201b1_2_input_info.npy", voxel_grid)

# # 输入和输出文件夹路径
# input_folder = "dataset_point_cloud_1109"
# output_folder = "dataset_voxel_1109"

# # 创建输出文件夹（如果不存在）
# os.makedirs(output_folder, exist_ok=True)

# # 遍历 input_folder 中的所有 .npy 文件
# for filename in os.listdir(input_folder):
#     if filename.endswith(".npy"):
#         # 构造完整路径
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)
        
#         # 加载点云数据
#         pointcloud = np.load(input_path)
        
#         # 转换为体素网格
#         voxel_grid = pointcloud_to_voxel_grid(pointcloud, voxel_size=0.05)
        
#         # 保存体素网格到输出文件夹
#         np.save(output_path, voxel_grid)
        
#         print(f"已处理并保存：{filename}")
        
# print("所有文件已处理完成！")
