import torch
import numpy as np
import os
import open3d as o3d
from torch.utils.data import DataLoader
# 注意改这里的路径
from model_training_1208 import ResNet, BasicBlock, VoxelDataset
import json

# 设置CUDA环境
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def save_npy_and_ply_from_voxels(voxel_data, save_npy_path, save_N_3_npy_path, save_ply_path, prefix, last_part, threshold=0.5):
    ray_points_camera = np.load("ray_points_camera_1208.npy")
    _, max_step, width, height = voxel_data.shape
    voxels_mask = voxel_data[0].cpu().detach().numpy()
    voxels_mask = voxels_mask.transpose(1, 2, 0)
    binary_voxels = (voxels_mask > threshold).astype(np.uint8)

    # 保存体素数据为 .npy 文件
    file_name = f"{prefix}_{last_part}_predicted.npy"
    np.save(os.path.join(save_npy_path, file_name), binary_voxels)

    voxels_mask = voxels_mask.reshape(-1, max_step)
    filtered_points = ray_points_camera[voxels_mask > threshold]

    # 保存点云为 .npy 文件
    point_cloud_npy_file = f"{prefix}_{last_part}_predicted_N_3.npy"
    np.save(os.path.join(save_N_3_npy_path, point_cloud_npy_file), filtered_points)

    # 保存点云为 .ply 文件
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    o3d.io.write_point_cloud(os.path.join(save_ply_path, f"{prefix}_{last_part}_predicted.ply"), pcd)

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
    # 指定单个测试文件路径
    test_input_file = "ex_test/0a761819-05d1-4647-889b-a726747201b1-copy_1207_input.npy"
    test_ground_truth_file = "ex_test/0a761819-05d1-4647-889b-a726747201b1-copy_1207_gt.npy"
    
    # 创建测试数据集和数据加载器
    test_dataset = VoxelDataset([test_input_file], [test_ground_truth_file])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载训练好的模型
    model = ResNet(BasicBlock, [2, 2, 2, 2]).cuda()
    
    checkpoint = torch.load('checkpoints/model_epoch_35_1208_93_85.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建保存目录
    save_npy_path = "single_predicted_npy"
    save_N_3_npy_path = "single_predicted_N_3_npy"
    save_ply_path = "single_redicted_ply"
    metrics_save_path = "single_model_metrics"
    os.makedirs(save_npy_path, exist_ok=True)
    os.makedirs(save_N_3_npy_path, exist_ok=True)
    os.makedirs(save_ply_path, exist_ok=True)
    os.makedirs(metrics_save_path, exist_ok=True)

    # 推理并保存结果
    with torch.no_grad():
        for input_info, ground_truth in test_loader:
            ground_truth = ground_truth.cuda()
            base_name = os.path.splitext(os.path.basename(test_input_file))[0]
            identifier = base_name.split('_')[0]
            last_part = base_name.split('_')[1]

            input_info = input_info.cuda()
            with torch.amp.autocast('cuda'):
                output = model(input_info)

            save_npy_and_ply_from_voxels(output[0], save_npy_path, save_N_3_npy_path, save_ply_path, identifier, last_part)

            # 计算指标
            precision, recall, iou = calculate_precision_recall(output, ground_truth)
            metrics = {"precision": precision, "recall": recall, "iou": iou}

            # 保存指标为 JSON 文件
            json_file_path = os.path.join(metrics_save_path, f"{identifier}_metrics.json")
            with open(json_file_path, 'w') as f:
                json.dump(metrics, f, indent=4)

    print("Processing complete. Results saved.")
