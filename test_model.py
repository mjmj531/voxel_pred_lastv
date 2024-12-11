###
# test_1111是所有提供测试的数据集
# checkpoint 在服务器上下载回来，放在checkpoints文件夹下
# 生成的是预测的点云文件，存在predicted_ply文件夹下
# 运行时需要修改模型路径，以及保存路径

import torch
import numpy as np
import os
import open3d as o3d
from torch.utils.data import DataLoader
from model_resnet_1111_depth_weight_and_gauss import ResNet, BasicBlock, VoxelDataset 
import matplotlib.pyplot as plt
import json

# 设置CUDA环境
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def save_npy_file(data, folder, scene, file_prefix, file_suffix):
    """保存.np文件, 文件名格式为: <scene>_<file_prefix>_<file_suffix>.npy"""
    file_name = f"{scene}_{file_prefix}_{file_suffix}.npy"
    file_path = os.path.join(folder, file_name)
    np.save(file_path, data.cpu().numpy())

def save_npy_and_ply_from_voxels(voxel_data, save_npy_path, save_N_3_npy_path, save_ply_path, prefix, last_part, threshold=0.5):
    """
    从体素数据保存点云。

    :param voxel_data: 体素数据，形状为 (1, max_step, width, height) max_step==depth
    :param save_path: 保存点云的路径
    :param threshold: 保存点云的体素阈值
    """
    ray_points_camera = np.load("ray_points_camera_1111.npy")
    _, max_step, width, height = voxel_data.shape
    print(f"depth: {max_step}, width: {width}, height: {height}")

    voxels_mask = voxel_data[0].cpu().detach().numpy()  # 转为 numpy 数组
    print(voxels_mask.shape) # (depth, width, height) (50, 100, 75)
    voxels_mask = voxels_mask.transpose(1, 2, 0)  # 转置，使 (width, height, depth) 顺序
    print("voxel_mask shape after transpose:  ",voxels_mask.shape) # (width, height, depth) (100, 75, 50)

    # 应用阈值，将 >= threshold 的值设置为 1，< threshold 的值设置为 0
    binary_voxels = (voxels_mask > threshold).astype(np.uint8)
    print("二值化后的体素网格形状:", binary_voxels.shape)
    # 直接在这里存npy文件
    file_name = f"{prefix}_{last_part}_predicted.npy"
    file_path = os.path.join(save_npy_path, file_name)
    np.save(file_path, binary_voxels)  # 保存体素数据为 .npy 文件
    print(f"Voxel data has been saved to {file_path}")  # 保存体素数据为 .npy 文件

    voxels_mask = voxels_mask.reshape(-1, max_step)
    print("voxel_mask shape after reshape: ",voxels_mask.shape) # (width*height, max_step) (7500, 50)
    filtered_points = ray_points_camera[voxels_mask > threshold]  # 过滤掉阈值以下的点

    # 保存点云为 .npy 文件
    point_cloud_npy_file = f"{prefix}_{last_part}_predicted_N_3.npy"
    point_cloud_npy_path = os.path.join(save_N_3_npy_path, point_cloud_npy_file)
    np.save(point_cloud_npy_path, filtered_points)  # 保存过滤后的点云为 .npy 文件
    print(f"Filtered point cloud has been saved to {point_cloud_npy_path}")

    # # 获取大于阈值的体素坐标
    # points = np.argwhere(voxels > threshold)  # 获取占用的体素坐标

    # # 将坐标转换为点云格式 (x, y, z)
    # point_cloud = points[:, [2, 1, 0]]  # 交换轴，使 (x, y, z) 顺序
    # # print(point_cloud.shape)

    # 创建 open3d 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # 保存为 PLY 文件，命名格式为：prefix_predicted.ply
    o3d.io.write_point_cloud(f"{save_ply_path}/{prefix}_{last_part}_predicted.ply", pcd)
    print(f"Point cloud has been saved to {save_ply_path}/{prefix}_{last_part}_predicted.ply")

def calculate_precision_recall(output, ground_truth, threshold=0.5):
    preds = (output > threshold).float()

    ground_truth = ground_truth.reshape(-1)
    preds = preds.reshape(-1)

    TP = (preds * ground_truth).sum().item()
    FP = (preds * (1 - ground_truth)).sum().item()
    FN = ((1 - preds) * ground_truth).sum().item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return precision, recall, iou

if __name__ == '__main__':
    # 设置测试数据集的文件路径
    ### 这里注意把test数据集从服务器下载回来

    test_input_files = [f"ex_test/{file_name}" for file_name in os.listdir('ex_test') if 'input' in file_name]
    test_ground_truth_files = [f"ex_test/{file_name}" for file_name in os.listdir('ex_test') if 'gt' in file_name]
    print(test_input_files)

    # test_input_files.sort()
    # test_ground_truth_files.sort()

    # # 截取前120个文件
    # test_input_files = test_input_files[:120]
    # test_ground_truth_files = test_ground_truth_files[:120]

    # 创建测试数据集和数据加载器
    test_dataset = VoxelDataset(test_input_files, test_ground_truth_files)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载训练好的模型
    model = ResNet(BasicBlock, [2, 2, 2, 2]).cuda()  # 每层的块数量
    ###########根据需要更改模型路径###########
    checkpoint = torch.load('checkpoints/model_epoch_28.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(torch.load('model_with_dataset_1030/model_epoch_1.pth', weights_only=True))  # 替换为模型保存路径
    model.eval()

    # 创建保存目录
    save_npy_path = "pred_outcome_for_npy_and_ply/predicted_npy_pw_0.64_origin_lr_0.0008_bs_16_epoch_28"
    save_N_3_npy_path = "pred_outcome_for_npy_and_ply/predicted_N_3_npy_pw_0.64_origin_lr_0.0008_bs_16_epoch_28"
    save_ply_path = "pred_outcome_for_npy_and_ply/predicted_ply_pw_0.64_origin_lr_0.0008_bs_16_epoch_28"
    metrics_save_path = "model_metrics_pw_0.64_origin_lr_0.0008_bs_16_epoch_28"  # 用于保存指标的目录
    os.makedirs(save_ply_path, exist_ok=True)
    os.makedirs(save_N_3_npy_path, exist_ok=True)
    os.makedirs(save_npy_path, exist_ok=True)
    os.makedirs(metrics_save_path, exist_ok=True)

    # 进行模型推理和可视化
    with torch.no_grad():
        for batch_idx, (input_info, ground_truth) in enumerate(test_loader):
            ground_truth = ground_truth.cuda()
            base_name = os.path.splitext(os.path.basename(test_input_files[batch_idx]))[0]
            
            identifier = base_name.split('_')[0]
            print(f"Predicting for {identifier}")
            last_part = base_name.split('_')[1]
            print(f"last_part: {last_part}")

            input_info = input_info.cuda()

            with torch.amp.autocast('cuda'):
                output = model(input_info)

            # 保存预测点云
            # save_npy_file(output[0], save_npy_path, f"{identifier}", f"{last_part}", "predicted")
            save_npy_and_ply_from_voxels(output[0], save_npy_path, save_N_3_npy_path, save_ply_path, f"{identifier}", f"{last_part}")

            # 计算精度、召回率和 IoU
            precision, recall, iou = calculate_precision_recall(output, ground_truth)

            metrics = {
                "precision": precision,  # 将张量转换为浮点数
                "recall": recall,
                "iou": iou
            }

            # 保存 JSON 文件
            json_file_path = os.path.join(metrics_save_path, f"aaa_metrics.json")
            with open(json_file_path, 'w') as f:
                json.dump(metrics, f, indent=4)

    print("Point clouds have been saved successfully.")


# def render_ply(ply_path, save_path=None):
#     # 加载PLY文件
#     point_cloud = o3d.io.read_point_cloud(ply_path)

#     # 放大物体（例如放大两倍）
#     scale_factor = 2.0
#     point_cloud.scale(scale_factor, center=point_cloud.get_center())

#     # 创建一个可视化窗口
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()

#     # 将点云添加到窗口
#     vis.add_geometry(point_cloud)

#     # 渲染并更新视图
#     vis.update_geometry(point_cloud)
#     vis.poll_events()
#     vis.update_renderer()

#     # 设置视图参数
#     view_control = vis.get_view_control()

#     # 设置相机位置（在物体的斜上方）
#     camera_position = np.array([-1.0, 1.0, 1.0])  # 根据物体的实际位置调整此位置
#     view_control.set_lookat(camera_position)  # 注视相机位置

#     # 设置相机注视的点（物体的中心）
#     lookat_point = np.array([0.0, 0.0, 0.0])  # 物体中心位置
#     view_control.set_lookat(lookat_point)

#     # 设置相机的前方方向（从相机位置指向注视点）
#     front_vector = lookat_point - camera_position  # 从相机位置指向注视点
#     front_vector /= np.linalg.norm(front_vector)  # 归一化
#     view_control.set_front(front_vector)

#     # 设置相机的上方向（通常为Y轴正方向）
#     up_vector = np.array([0.0, 1.0, -1.0])  # Z轴正方向
#     view_control.set_up(up_vector)

#     # 更新视图
#     vis.update_geometry(point_cloud)
#     vis.poll_events()
#     vis.update_renderer()

#     # 截图并保存
#     vis.capture_screen_image(f"{save_path}_screenshot.png")

#     # 关闭可视化窗口
#     vis.destroy_window()

# def visualize_point_clouds(input_path, ground_truth_path, output_path, metrics, save_dir, prefix, last_part):
#     """
#     将输入、预测输出和真实值的点云进行可视化并保存为图片。
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     paths = [input_path, ground_truth_path, output_path]
#     titles = ["Input", "Ground Truth", "Output"]
#     iou, precision, recall = metrics

#     for i, (path, title) in enumerate(zip(paths, titles)):
#         img_save_path = os.path.join(save_dir, f"{prefix}_{title.lower()}.png")
#         print("path: ", path)
#         render_ply(path, save_path=img_save_path)

#         # img = plt.imread(img_save_path)
#         # axes[i].imshow(img)
#         axes[i].axis("off")
#         axes[i].set_title(f"{title}\nIoU: {iou:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

#     plt.savefig(os.path.join(save_dir, f"{prefix}_{last_part}_visualization.png"))
#     # plt.show()
#     plt.close()
#     print(f"Visualization saved at {os.path.join(save_dir, f'{prefix}_{last_part}_visualization.png')}")