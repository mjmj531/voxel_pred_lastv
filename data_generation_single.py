import torch
import numpy as np
# import matplotlib.pyplot as plt
import json
import time
import os
import open3d as o3d
import numpy as np
# from data_generation_tramsform import *
from data_generation_tramsform_1208 import *

# 从随机生成的数据集中读取外参矩阵
def load_matrix_from_json(file_path):
    """
    读取 JSON 文件并转换为 PyTorch 张量。

    参数:
        file_path: str，JSON 文件的路径。
    
    返回:
        extrinsics: torch.Tensor，形状为 (4, 4) 的矩阵。
    """
    with open(file_path, 'r') as f:
        # 加载 JSON 文件内容
        matrix_data = json.load(f)
    
    # 转换为 PyTorch 张量并移动到 GPU
    extrinsics = torch.tensor(matrix_data).float().cuda()
    
    return extrinsics

def load_camera_extrinsics(file_path):

    with open(file_path, 'r') as f:
        camera_data = json.load(f)

    # 提取内参矩阵、分辨率
    extrinsics = torch.tensor(camera_data['extrinsics']).float().cuda()

    return extrinsics

if __name__ == '__main__':

    # 调试单个场景
    step_size = 0.04
    occupancy_grid = np.load('/home/mj/Desktop/newversion/Scene_new/0a761819-05d1-4647-889b-a726747201b1-copy.npy')
    occupancy_grid = torch.tensor(occupancy_grid, dtype=torch.float16).cuda()
    extrinsics = load_camera_extrinsics('param_for_debug_from_blender/tmp_tmp_ex.json')
    print("extrinsics: ", extrinsics)
    intrinsics, image_width, image_height = load_camera_intrinsics_resolution(
        'intrinsics_and_resolution.json'
    )
    ray_points_camera, max_steps = save_ray_points_camera_nv(image_width, image_height, intrinsics, 'param_for_debug_from_blender/ray_points_camera_test.npy', step_size=step_size)
    gt, input = ray_march_parallel(occupancy_grid, image_width, image_height, intrinsics, extrinsics, ray_points_camera, max_steps, step_size=step_size)
    print(gt.shape)
    print(input.shape)
    np.save('ex_test/0a761819-05d1-4647-889b-a726747201b1-copy_1207_input.npy', input.cpu().numpy())
    np.save('ex_test/0a761819-05d1-4647-889b-a726747201b1-copy_1207_gt.npy', gt.cpu().numpy())
  
    save_npy_to_ply(ray_points_camera, gt, 'ex_test','0a761819-05d1-4647-889b-a726747201b1-copy_1207_gt.ply')
