import open3d as o3d
import numpy as np


# (N,3)
# points = np.load('pred_outcome_for_npy_and_ply/predicted_N_3_npy_pw_0.64_origin_lr_0.0008_bs_16_epoch_28/a5af701c-1825-40ce-a9d7-0b7ed08fb43b_4_predicted_N_3.npy')

points = np.load("param_for_debug_from_blender/ray_points_camera_test.npy")
print(points.shape)
points = points.reshape(-1, 3)
print(points.dtype)

# 加载 ray_points 和 gt_mask
# ray_points = np.load("knn_test/regular_pointcloud.npy")  # 形状为 (N, max_step, 3)
# gt_mask = np.load("dataset_1111/0a761819-05d1-4647-889b-a726747201b1_100_input_info.npy")  # 形状为 (N, max_step)
# print(gt_mask.shape)
# gt_mask = gt_mask.reshape(-1, 50)
# print(gt_mask.shape)

# 根据 gt_mask 筛选出有效的点
# 将 gt_mask 扁平化，使得我们能直接按位置筛选
# filtered_points = ray_points[gt_mask == 1]

# 创建 open3d 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 可视化点云
o3d.visualization.draw_geometries([pcd])
