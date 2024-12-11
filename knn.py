import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
import os

# 但是现在是用的voxel掩码当作已知的，而不是点云数据，所以需要修改代码以适应新的输入格式。
# def voxel_knn(voxel, resolution, k=5):
#     # # 获取体素网格中的所有点（离散）
#     # occupied_points = np.argwhere(voxel == 1)  # 获取占用点的索引

#     # 获取所有点的坐标（用于构建KDTree）
#     all_voxel_points = np.argwhere(np.ones_like(voxel))  # 获取所有点的坐标

#     # 将离散网格转换为更密集的网格
#     grid_shape = voxel.shape
#     x = np.arange(0, grid_shape[0])
#     y = np.arange(0, grid_shape[1])
#     z = np.arange(0, grid_shape[2], resolution)
#     grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
#     all_points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1)

#     # 使用KDTree查找最近邻点
#     kdtree = KDTree(all_voxel_points)  # 构建占用点的KDTree，只存位置不存状态
#     _, indices = kdtree.query(all_points, k) 

#     # 对最近邻点进行投票
#     neighbor_coords = all_voxel_points[indices]  # 最近邻的点坐标, shape (N_new, k, 3)
#     # 取出邻域内的点的状态，shape (N_new, k)
#     # x_coords = neighbor_coords[:, :, 0]
#     # y_coords = neighbor_coords[:, :, 1]
#     # z_coords = neighbor_coords[:, :, 2]
#     # neighbor_values = voxel[x_coords, y_coords, z_coords]
#     neighbor_values = voxel[tuple(neighbor_coords.transpose(2, 0, 1))]
#     votes = np.mean(neighbor_values, axis=1)  # 获取邻域内的平均值
#     interpolated_values = (votes > 0.5).astype(int)  # 投票阈值 > 0.5 即认为是占用, shape (N_new,)

#     # 将插值后的值重新组合成三维空间
#     interpolated_grid = interpolated_values.reshape(grid_x.shape)
#     return interpolated_grid

# 使用所有点云数据和占用掩码对查询点进行 KNN 插值预测
def voxel_knn_and_save_as_ply(points, occupancy_mask, query_points, k=5, save_path=None):
    """
    使用所有点云数据和占用掩码对查询点进行 KNN 插值预测。
    参数:
        points: numpy.ndarray, 所有的点云数据 (width*height, depth, 3) 形式
        occupancy_mask: numpy.ndarray, 占用掩码，标记哪些点是占用的 (width, height, depth) 形式
        query_points: numpy.ndarray, 查询点云数据 (M, 3) 形式
        k: int, 最近邻点的数量
    返回:
        predictions: numpy.ndarray, 预测的结果 (M,)，每个点是占用 (1) 还是非占用 (0)
    """
    # 展平 points 到 (N, 3) 形式
    flattened_points = points.reshape(-1, 3)  # (7500 * 50, 3)
    print("flattened_points.shape: ", flattened_points.shape)  # (7500 * 50, 3)
    
    # 展平 occupancy_mask 到 (N,) 形式
    flattened_mask = occupancy_mask.ravel()  # (100 * 75 * 50,)
    # print("flattened_mask.shape: ", flattened_mask.shape)  # (100 * 75 * 50,)

    # 使用所有点来初始化 KDTree
    kdtree = KDTree(flattened_points)  # 使用所有点云数据构建 KDTree
    print("kdtree构建完成")  # (7500 * 50, 3)
    
    # 使用 KDTree 查找最近邻点
    _, indices = kdtree.query(query_points, k)  # 找到每个查询点的最近 k 个点
    print("indices.shape: ", indices.shape)  # (M, k)

    # 对最近邻点进行投票 (简单二值投票)
    # 获取最近邻的掩码，shape (M, k)
    neighbor_mask = flattened_mask[indices]  
    
    # 对于每个查询点，计算其邻居中有多少个是占用点
    votes = np.sum(neighbor_mask, axis=1)  # 计算每个查询点的投票数

    # 如果投票数超过阈值，则认为是占用 (阈值是 k//2)
    predictions = (votes >= (k // 2)).astype(int)  # 投票阈值 > k//2 即认为是占用

    # 保存预测为点云
    if save_path:
        occupied_points = query_points[predictions == 1]  # 提取被预测为占用的点
        save_point_cloud_open3d(occupied_points, save_path)

    return predictions


def save_point_cloud_open3d(points, file_path):
    """
    使用 Open3D 将点云数据保存为 .ply 文件。
    
    参数:
        points: numpy.ndarray, 点云数据 (N, 3)
        file_path: str, 保存文件的路径
    """
    # 创建 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    
    # 将点云数据转换为 Open3D 格式
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # 保存点云为 PLY 文件
    o3d.io.write_point_cloud(file_path, point_cloud)
    print(f"Point cloud saved to {file_path}")

# 加载PLY文件并转换为NumPy数组
def load_ply_as_numpy(ply_file_path):
    """
    从 .ply 文件加载点云并转换为 NumPy 数组。
    参数:
        ply_file_path: str, .ply 文件路径
    返回:
        points: numpy.ndarray, 点云坐标 (N, 3)
    """
    # 使用 Open3D 读取 .ply 文件
    point_cloud = o3d.io.read_point_cloud(ply_file_path)
    
    # 获取点云的点坐标
    points = np.asarray(point_cloud.points)
    print("points.shape: ", points.shape)
    return points


if __name__ == "__main__":
    points = np.load("ray_points_camera_1111.npy") #(7500,50,3)
    # 改
    occupancy_mask = np.load("pred_outcome_for_npy_and_ply/predicted_npy_pw_0.64_origin_lr_0.0008_bs_16_epoch_28/a7a7529d-7799-4213-9ed2-384be3540fe1_126_predicted.npy") #(7500,50)
    print("占用掩码的形状:", occupancy_mask.shape) #(100,75,50)
    print("占用掩码的和:", occupancy_mask.sum())
    query_points = np.load("knn_test/roi_cube_from_ray_to_npy.npy")
    print("查询点云数据的形状:", query_points.shape)
    # 改
    save_path = "knn_test/a7a7529d-7799-4213-9ed2-384be3540fe1_126_query_points_roi.ply"
    predictions_query = voxel_knn_and_save_as_ply(points, occupancy_mask, query_points, save_path=save_path)
    print("预测结果为1的点的数量:", predictions_query.sum())


# # eg
# voxel = np.load("pred_outcome_for_npy_and_ply/predicted_npy_pw_0.64_origin_lr_0.0008_bs_16_epoch_28/a5af701c-1825-40ce-a9d7-0b7ed08fb43b_81_predicted.npy")
# print("原始体素网格的形状:", voxel.shape)
# print("数据类型:", voxel.dtype)
# print("最小值:", voxel.min())
# print("最大值:", voxel.max())
# print("是否有无穷值:", np.isinf(voxel).any())
# print("是否有NaN值:", np.isnan(voxel).any())
# print(voxel.sum())
# # 只对深度进行插值
# interpolated_voxel = voxel_knn(voxel, resolution=0.5)
# np.save("knn_test/a5af701c-1825-40ce-a9d7-0b7ed08fb43b_81_predicted_knn.npy", interpolated_voxel)

# ray_points_camera = np.load("ray_points_camera_test_knn.npy")
# print("相机空间射线点云的形状:", ray_points_camera.shape)

# save_npy_to_ply(ray_points_camera, interpolated_voxel, "knn_test", "a5af701c-1825-40ce-a9d7-0b7ed08fb43b_81_predicted_knn.ply")

# print("插值后连续网格的形状:", interpolated_voxel.shape)
# print(interpolated_voxel.sum())
