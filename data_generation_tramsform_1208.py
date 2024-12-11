# [把生成ray_points_camera的函数改对了（按直角边）

import torch
import numpy as np
# import matplotlib.pyplot as plt
import json
import time
import os
import open3d as o3d
import numpy as np
from extrincis_generate import *

def load_camera_intrinsics_resolution(file_path):

    with open(file_path, 'r') as f:
        camera_data = json.load(f)

    # 提取内参矩阵、分辨率
    intrinsics = torch.tensor(camera_data['intrinsics']).float().cuda()
    image_width = camera_data['image_resolution']['width']
    image_height = camera_data['image_resolution']['height']

    return intrinsics, image_width, image_height

def load_camera_extrinsics():
    extrinsics = generate_random_camera_extrinsics()
    extrinsics = torch.tensor(extrinsics).float().cuda()
    return extrinsics

def load_camera_extrinsics_from_blender(file_path):
    with open(file_path, 'r') as f:
        camera_data = json.load(f)
    extrinsics = torch.tensor(camera_data['extrinsics']).float().cuda()
    return extrinsics

def pixel_to_camera_coords_all(pixels, intrinsics_inv):
    """
    像素坐标批量转换为相机坐标系中的坐标
    """
    camera_coords = intrinsics_inv @ pixels.T
    # 返回形状为 (3, N) 的张量
    # 这里更改了正负号
    return torch.cat([camera_coords[0:1], camera_coords[1:2], -torch.ones(1, camera_coords.shape[1]).cuda()])
    # return torch.cat([camera_coords[0:1], -camera_coords[1:2], -torch.ones(1, camera_coords.shape[1]).cuda()])

def normalize_vector_all(vec):
    """
    对向量进行归一化处理（并行版本）
    vec: 形状为 (3, N) 的张量，表示 N 个光线方向
    """
    # 计算每个向量的范数（按列计算）
    norm = vec.norm(dim=0, keepdim=True)
    
    # 避免除以 0
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)

    # 对每个向量进行归一化
    return vec / norm

def camera_to_world_coords_all(ray_direction_camera, extrinsics_inv):
    """
    将相机坐标系中的光线方向转换为世界坐标系中的方向（并行版本）
    ray_direction_camera: 形状为 (3, N) 的张量，表示 N 个光线方向
    extrinsics_inv: 外参矩阵的逆，形状为 (4, 4)
    return 形状为 (3, N) 的张量，表示 N 个光线方向
    """
    # 提取外参矩阵的旋转部分
    rotation_matrix = extrinsics_inv[:3, :3]

    # 批量转换光线方向
    ray_direction_world = rotation_matrix @ ray_direction_camera  # 矩阵乘法
    return normalize_vector_all(ray_direction_world)

def world_to_camera_coords_all(ray_direction_world, extrinsics):
    """
    将世界坐标系中的光线方向转换为相机坐标系中的方向（并行版本）
    ray_direction_world: 形状为 (3, N) 的张量，表示 N 个光线方向
    extrinsics: 外参矩阵，形状为 (4, 4)
    return 形状为 (3, N) 的张量，表示 N 个光线方向
    """
    # 提取外参矩阵的旋转部分
    rotation_matrix = extrinsics[:3, :3]  # 提取旋转矩阵
    
    # 批量转换光线方向
    ray_direction_camera = rotation_matrix.t() @ ray_direction_world  # 旋转矩阵的转置应用于光线方向
    return ray_direction_camera

def world_to_camera_ray_all(ray_direction_world, extrinsics):
    """
    将世界坐标系中的光线方向转换为相机坐标系中的方向（并行版本）
    ray_direction_world: 形状为 (N, max_steps, 3) 的张量，表示 N 条光线的方向
    extrinsics: 外参矩阵，形状为 (4, 4)
    return 形状为 (N, max_steps, 3) 的张量，表示 N 条光线的方向
    """
    # 提取外参矩阵的旋转部分
    rotation_matrix = extrinsics[:3, :3]  # 提取旋转矩阵

    # 扩展 ray_direction_world 的维度，使其适配矩阵乘法
    # ray_direction_world: (N, max_steps, 3) -> (N * max_steps, 3)
    ray_direction_world_flat = ray_direction_world.reshape(-1, 3)

    # 批量转换光线方向
    # 使用转置矩阵进行乘法
    ray_direction_camera_flat = torch.matmul(rotation_matrix.t(), ray_direction_world_flat.t()).t()

    # 将结果恢复为 (N, max_steps, 3)
    ray_direction_camera = ray_direction_camera_flat.view(ray_direction_world.shape[0], ray_direction_world.shape[1], 3)

    return ray_direction_camera

# def save_ray_points_camera(image_width, image_height, intrinsics, file_path, step_size, deep = 2.0):
#     ############ 限制深度为2m ###########
#     max_steps = int(deep / step_size)
    
#     # 创建像素网格，i表示行，j表示列
#     i, j = torch.meshgrid(torch.arange(image_height, device='cuda'),
#                           torch.arange(image_width, device='cuda'),
#                           indexing='ij')
#     # 这里先j再i
#     pixels = torch.stack([j.float() + 0.5, i.float() + 0.5, torch.ones_like(i).float().cuda()], dim=-1).reshape(-1, 3)
#     intrinsics_inv = torch.inverse(intrinsics)
#     ray_direction_camera = normalize_vector_all(pixel_to_camera_coords_all(pixels, intrinsics_inv))
#     # 在相机坐标系下生成所有步长位置
#     t = torch.arange(0, max_steps).float().cuda() * step_size
#     t = t.unsqueeze(0).unsqueeze(2)  # (1, max_steps, 1)
#     # 把ray_points_camera 单独存下来
#     ray_points_camera = ray_direction_camera.t().unsqueeze(1) * t  # (N, max_steps, 3)
#     np.save(file_path, ray_points_camera.cpu().numpy())
#     return ray_points_camera, max_steps

def save_ray_points_camera_nv(image_width, image_height, intrinsics, file_path, step_size, depth=2.0):
    """
    保存每条光线在相机坐标系下的点云，每条光线上取 num_points 个点，
    每个点对应的深度等间距，总深度为 depth。
    """
    # 计算每条光线的点数
    num_points = int(depth / step_size) #即为max_steps
    
    # 创建像素网格，i表示行，j表示列
    i, j = torch.meshgrid(torch.arange(image_height, device='cuda'),
                          torch.arange(image_width, device='cuda'),
                          indexing='ij')
    
    # 每个像素的归一化坐标 (u, v)
    pixels = torch.stack([j.float() + 0.5, i.float() + 0.5, torch.ones_like(i).float().cuda()], dim=-1).reshape(-1, 3)
    
    # 计算相机坐标系下的光线方向
    intrinsics_inv = torch.inverse(intrinsics)
    ray_direction_camera = normalize_vector_all(pixel_to_camera_coords_all(pixels, intrinsics_inv))

    # 对每条光线生成等深度间隔的点
    z_values = torch.linspace(0, -depth, steps=num_points, device='cuda')  # 深度等间隔 (num_points,)
    z_values = z_values.unsqueeze(0).unsqueeze(2)  # (1, num_points, 1)
    
    # 根据深度计算点云位置
    # 使用深度z_values计算 x, y 坐标：x = z * ray_x / ray_z, y = z * ray_y / ray_z
    ray_directions = ray_direction_camera.t().unsqueeze(1)  # (N, 1, 3)
    ray_z = ray_directions[:, :, 2:3]  # 提取 z 分量 (N, 1, 1)
    ray_points_camera = ray_directions / ray_z * z_values  # (N, num_points, 3)

    # 保存点云数据
    np.save(file_path, ray_points_camera.cpu().numpy())
    
    return ray_points_camera, num_points


def ray_march_parallel(occupancy_grid, image_width, image_height, intrinsics, extrinsics, ray_points_camera, max_steps, step_size, grid_size = [0.02, 0.02, 0.02], deep = 2.0):
    # 相机位置
    camera_position = extrinsics[:3, 3].unsqueeze(0)
    x, y, z = camera_position[0, 0], camera_position[0, 1], camera_position[0, 2]
    
    # 将相机位置映射到 occupancy grid 的索引上
    x_index = ((x + 3) / grid_size[0]).int()
    y_index = (y / grid_size[1]).int()
    z_index = ((z + 4) / grid_size[2]).int()
    
    # print("x_index, y_index, z_index: ", x_index, y_index, z_index)

    # 判断相机位置是否在occupancy grid范围内，且向下y方向投影是否为1
    # if 0 <= x_index < occupancy_grid.shape[0] and 0 <= z_index < occupancy_grid.shape[2]: # 设置相机参数时已经实现 条件必然成立
    # 从相机位置 y_index 到 0 的所有 y 值（相机正下方的体素） 稍微允许一些富裕
    y_slice = occupancy_grid[x_index-1: x_index+2, :y_index+1, z_index-1: z_index+2]

    # filter
    ## 但是理论上这个问题不应该有，因为我看(300,100,400)的最底层都有1
    if not torch.any(y_slice == 1):
        # 如果没有1，说明相机不在在室内，返回 None
        print("Camera not inside the room")
        return None, None
    
    if torch.all(y_slice == 1):
        print("camera might inside the object")
        return None, None
    
    ############ 限制深度为2m ###########
    
    # 将相机坐标系下的点转换到世界坐标系
    extrinsics_inv = torch.inverse(extrinsics)
    ray_points_world_relative = torch.matmul(ray_points_camera, extrinsics_inv[:3, :3].T) 
    ray_points_world_positions = camera_position + ray_points_world_relative # (N, max_steps, 3)

    # 计算occu_grid的占用情况
    grid_x = ((ray_points_world_positions[..., 0] + 3) / grid_size[0]).int()
    grid_y = (ray_points_world_positions[..., 1] / grid_size[1]).int()
    # print("grid_y.shape: ", grid_y.shape) # (N, max_steps)
    grid_z = ((ray_points_world_positions[..., 2] + 4) / grid_size[2]).int()

    # 检查每个位置是否在 grid 内
    in_grid_mask = (0 <= grid_x) & (grid_x < occupancy_grid.shape[0]) & \
                (0 <= grid_y) & (grid_y < occupancy_grid.shape[1]) & \
                (0 <= grid_z) & (grid_z < occupancy_grid.shape[2])
    # 将无效的 grid 索引设为 -1（避免索引错误）
    grid_x = torch.where(in_grid_mask, grid_x, torch.tensor(-1, device='cuda'))
    grid_y = torch.where(in_grid_mask, grid_y, torch.tensor(-1, device='cuda'))
    grid_z = torch.where(in_grid_mask, grid_z, torch.tensor(-1, device='cuda'))
    # 把hitsreshap为(image_width, image_height, max_steps)即为ground truth
    hits = occupancy_grid[grid_x, grid_y, grid_z] == 1 

    # filter
    if not torch.any(hits):
        # 没有击中任何目标，返回 None，None
        print("No target hit!")
        return None, None
    ###############
    ground_truth = hits.to(torch.float16) #（N, max_steps）

    # Calculate first hit index, but only set for rays that hit at least one voxel
    first_hit_indices = torch.argmax(hits.int(), dim=1)

    # Mask for rays with no hits (all False across max_steps)
    no_hits_mask = ~torch.any(hits, dim=1)

    # Set first_hit_indices to -1 for rays with no hits
    first_hit_indices = torch.where(no_hits_mask, torch.tensor(-1, device='cuda'), first_hit_indices)
    
    # filter
    # 将 -1 替换为一个较大的值，例如很大的整数 64
    masked_indices = torch.where(first_hit_indices == -1, torch.tensor(64, device='cuda'), first_hit_indices)
    if masked_indices.min() < 2:
        # 大概率相机在物体内部
        print("Camera inside the object")
        return None, None

    # 创建与hits相同形状的掩码，初始化为全零。基于first_hit_indices
    input_info = torch.zeros_like(ground_truth, dtype=torch.float16, device='cuda')

    # 对于每条光线，如果第一次击中位置索引为-1，则不设置任何位置
    valid_hits = first_hit_indices >= 0  # 找到有效的击中位置（即first_hit_indices >= 0）

    # 仅针对有效击中的光线进行索引
    valid_ray_indices = torch.arange(hits.shape[0], device='cuda')[valid_hits]
    valid_first_hit_indices = first_hit_indices[valid_hits]

    # 将有效击中位置的值设为1
    input_info[valid_ray_indices, valid_first_hit_indices] = 1 #（N, max_steps）

    return ground_truth.view(image_width, image_height, max_steps), input_info.view(image_width, image_height, max_steps)

def save_extrinsics_as_json(extrinsics, data_folder, file_suffix):
    # 将外参矩阵转换为 Python 可序列化的列表形式
    extrinsics_list = [[float(value) for value in row] for row in extrinsics]
    
    # 构造文件路径
    json_filename = os.path.join(data_folder, f"extrinsics_{file_suffix}.json")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
    
    # 保存为 JSON 文件
    with open(json_filename, 'w') as json_file:
        json.dump(extrinsics_list, json_file, indent=4)

def save_pointcloud_to_ply(pointcloud, save_path, filename):
    """
    将三维点云数据保存为 PLY 格式
    :param pointcloud: Tensor, 点云数据，形状为 (N, 3)
    :param filename: str, 保存的文件名
    """
    # 确保点云数据在CPU上并转换为numpy格式
    pointcloud_np = pointcloud.cpu().numpy()
    if pointcloud_np.size == 0:
        print("Warning: Point cloud data is empty.")
        return
    # 创建open3d点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_np)

    # 保存为PLY文件
    o3d.io.write_point_cloud(os.path.join(save_path, filename), pcd)

# 结合ray_points_camera生成点云
def save_npy_to_ply(ray_points_camera, mask, save_path, filename):
    """
    将三维点云数据保存为 PLY 格式
    :param pointcloud: Tensor, 点云数据，形状为 (N, 3)
    :param filename: str, 保存的文件名
    """
    ray_points_camera = ray_points_camera.cpu().numpy() # (N, max_steps, 3)
    _, max_steps, _ = ray_points_camera.shape
    # 确保点云数据在CPU上并转换为numpy格式
    ply_mask = mask.cpu().numpy() #(width, height, max_steps)
    ply_mask = ply_mask.reshape(-1, max_steps) # (width*height=N, max_steps)
    if ply_mask.size == 0:
        print("Warning: Point cloud data is empty.")
        return

    filtered_points = ray_points_camera[ply_mask == 1]
    # print("filtered_points.shape: ", filtered_points.shape)
    # 创建open3d点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # 保存为PLY文件
    o3d.io.write_point_cloud(os.path.join(save_path, filename), pcd)

# 特定格式保存.npy文件
def save_npy_file(data, folder, scene, file_prefix, file_suffix):
    """保存.np文件, 文件名格式为: <scene>_<file_prefix>_<file_suffix>.npy"""
    file_name = f"{scene}_{file_prefix}_{file_suffix}.npy"
    file_path = os.path.join(folder, file_name)
    np.save(file_path, data.cpu().numpy())

def generate_and_save_valid_extrinsics(occupancy_grid, num, intrinsics, image_width, image_height, data_folder, extrinsics_folder, scene, step_size):
    # max_steps = int(deep / step_size)
    ray_points_camera, max_steps = save_ray_points_camera_nv(image_width, image_height, intrinsics, 'ray_points_camera_vertical.npy', step_size=step_size)
    cnt = 0
    while cnt < num:
        extrinsics = load_camera_extrinsics()
        ground_truth, input_info = ray_march_parallel(
            occupancy_grid, image_width, image_height, intrinsics, extrinsics, ray_points_camera, max_steps, step_size
        )
        
        if input_info is not None and ground_truth is not None:
            file_suffix = f"{cnt}"
            # 保存外参矩阵为 JSON 文件
            save_extrinsics_as_json(extrinsics, extrinsics_folder, file_suffix)

            # 保存 input_info 和 ground_truth 为 npy 文件
            save_npy_file(input_info, data_folder, scene, file_suffix, 'input_info')
            save_npy_file(ground_truth, data_folder, scene, file_suffix, 'ground_truth')

            # 保存点云
            gt_ply_folder = 'gt_ply_1210'
            os.makedirs(gt_ply_folder, exist_ok=True)
            input_ply_folder = 'input_ply_1210'
            os.makedirs(input_ply_folder, exist_ok=True)

            save_npy_to_ply(ray_points_camera, input_info, input_ply_folder, f"{scene}_{file_suffix}_input_info.ply")
            save_npy_to_ply(ray_points_camera, ground_truth, gt_ply_folder, f"{scene}_{file_suffix}_ground_truth.ply")

            cnt += 1

# 在这里默认初始化step_size为0.04
def process_all_scenes(data_folder, scene_folder, step_size = 0.04, num = 128):
    # grid_size = [0.02, 0.02, 0.02]
    files = os.listdir(scene_folder)
    files.sort()

    # 加载相机的内参和分辨率
    intrinsics, image_width, image_height = load_camera_intrinsics_resolution(
        'intrinsics_and_resolution.json'
    )
    # 遍历 Scene_new 文件夹中的所有 .npy 文件
    for scene_file in files:
        if scene_file.endswith('.npy'):
            # 提取场景名称
            scene = os.path.splitext(scene_file)[0]
            print(f"Processing scene: {scene}")
            
            # 加载场景的占用网格
            scene_occu = os.path.join(scene_folder, scene)
            occupancy_grid = np.load(scene_occu + '.npy')
            occupancy_grid = torch.tensor(occupancy_grid, dtype=torch.float16).cuda()

            # 设置相机外参文件夹路径
            extrinsics_folder = os.path.join('cam_extrinsics_param_1210', scene)
            os.makedirs(extrinsics_folder, exist_ok=True)

            # 生成并保存有效的外参矩阵
            generate_and_save_valid_extrinsics(
                occupancy_grid, num, intrinsics, image_width, image_height,
                data_folder, extrinsics_folder, scene, step_size=step_size
            )

if __name__ == '__main__':

    Sceme_new_folder = '../Scene_new'
    step_size = 0.04

    data_folder = "dataset_1210"
    os.makedirs(data_folder, exist_ok=True)

    process_all_scenes(data_folder, Sceme_new_folder, step_size=step_size, num = 128)

