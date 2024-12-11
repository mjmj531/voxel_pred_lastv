# depth_weight_and_gauss_and_posweight
# 10：30 model_chechpoints_1113 model_epoch_4.pth

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader

import json
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # 使输入和输出通道一致
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # 残差连接
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 1  # 初始通道数
        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1])
        self.layer3 = self._make_layer(block, 64, num_blocks[2])
        self.layer4 = self._make_layer(block, 128, num_blocks[3])
        self.fc = nn.Conv3d(128, 1, kernel_size=3, padding=1)  # 最终输出层

    def _make_layer(self, block, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(x)
        return x


# 定义数据集

# 增加高斯噪声
class VoxelDataset(Dataset):
    def __init__(self, input_files, ground_truth_files, noise_std=0.04):
        self.input_files = input_files
        self.ground_truth_files = ground_truth_files
        self.noise_std = noise_std  # 噪声的标准差

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_data = np.load(self.input_files[idx])
        ground_truth_data = np.load(self.ground_truth_files[idx])

        # 转换为 PyTorch 张量
        input_tensor = torch.tensor(input_data, dtype=torch.float16).unsqueeze(0)  # (1, width, height, depth)
        ground_truth_tensor = torch.tensor(ground_truth_data, dtype=torch.float16).unsqueeze(
            0)  # (1, width, height, depth)

        # 添加高斯噪声
        noise = torch.randn_like(input_tensor) * self.noise_std  # 使用正态分布噪声
        input_tensor = input_tensor + noise  # 将噪声添加到输入数据中

        return input_tensor.permute(0, 3, 1, 2), ground_truth_tensor.permute(0, 3, 1, 2)


def calculate_precision_recall(output, ground_truth, threshold=0.5):
    preds = (output > threshold).float()

    ground_truth = ground_truth.reshape(-1)
    preds = preds.reshape(-1)

    TP = (preds * ground_truth).sum().item()
    FP = (preds * (1 - ground_truth)).sum().item()
    TN = ((1 - preds) * (1 - ground_truth)).sum().item()
    FN = ((1 - preds) * ground_truth).sum().item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return precision, recall, iou


if __name__ == '__main__':
    # 划分训练集和测试集#
    # 1208 version 只有训练集和测试集，没有验证集，因此val处用的是test
    train_input_files = [f"dataset_split_1208_local/train/{file_name}" for file_name in
                         os.listdir('dataset_split_1208_local/train') if 'input_info' in file_name]
    train_ground_truth_files = [f"dataset_split_1208_local/train/{file_name}" for file_name in
                                os.listdir('dataset_split_1208_local/train') if 'ground_truth' in file_name]

    train_input_files.sort()
    train_ground_truth_files.sort()

    val_input_files = [f"dataset_split_1208_local/test/{file_name}" for file_name in
                       os.listdir('dataset_split_1208_local/test') if 'input_info' in file_name]
    val_ground_truth_files = [f"dataset_split_1208_local/test/{file_name}" for file_name in
                              os.listdir('dataset_split_1208_local/test') if 'ground_truth' in file_name]

    val_input_files.sort()
    val_ground_truth_files.sort()

    # 创建数据加载器
    train_dataset = VoxelDataset(train_input_files, train_ground_truth_files)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

    val_dataset = VoxelDataset(val_input_files, val_ground_truth_files)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True)

    # 初始化模型
    model = ResNet(BasicBlock, [2, 2, 2, 2]).cuda()  # 每层的块数量
    # 初始化 wandb
    wandb.init(project="voxel_pred_1208", entity="ma-j22-thu")  # 请替换为您的项目名称和用户名

    # 记录超参数
    config = wandb.config
    config.epochs = 50
    config.learning_rate = 0.0008
    config.positive_weight = 0.64

    # 定义损失函数和优化器 (使用 BCEWithLogitsLoss)
    ### 确定 pos_weight ###
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.positive_weight).cuda())
    # criterion = nn.BCEWithLogitsLoss()

    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 使用 StepLR 调度器，每 1 个 epoch 调整学习率为原来的 0.9 倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # 创建 AMP 相关的 scaler
    scaler = torch.amp.GradScaler('cuda')

    for input_info, _ in train_loader:
        # 获取第三维depth大小
        depth_size = input_info.shape[2]
        print(f"Input shape: {depth_size}", flush=True)  # Input shape: torch.Size([8, 1, 50, 100, 75])
        break  # 只需获取一次，所以可以退出循环
    # 训练模型
    num_epochs = config.epochs
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (input_info, ground_truth) in enumerate(train_loader):
            input_info = input_info.cuda()
            ground_truth = ground_truth.cuda()

            ### 每一个batch更新一次 ###
            optimizer.zero_grad()

            # 使用 autocast 进行前向传播
            with torch.amp.autocast('cuda'):
                output = model(input_info)
                # loss = criterion(output, ground_truth)
                # 前半部分的损失 (batch_size, 1, depth//2, height, width)
                loss_front = criterion(output[:, :, :depth_size // 2, :, :],
                                       ground_truth[:, :, :depth_size // 2, :, :])

                # 后半部分的损失 (batch_size, 1, depth//2, height, width)
                loss_back = criterion(output[:, :, depth_size // 2:, :, :],
                                      ground_truth[:, :, depth_size // 2:, :, :])

                # 计算加权后的平均损失
                weighted_loss = 2.0 * loss_front.mean() + loss_back.mean()

            # 记录当前 batch 的加权训练损失
            wandb.log({"batch_train_loss": weighted_loss.item()})

            print(f"Train Loss (Batch {batch_idx + 1}/{len(train_loader)}): {weighted_loss.item():.4f}", flush=True)

            # 记录 epoch 累积的加权损失
            epoch_loss += weighted_loss.item()

            # 使用加权损失进行反向传播
            scaler.scale(weighted_loss).backward()  # 计算梯度

            ### 每一个batch更新一次 ###
            # 更新参数
            scaler.step(optimizer)
            scaler.update()

        # 打印每个周期的平均损失
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}", flush=True)
        wandb.log({"epoch_train_loss": avg_loss})

        # 验证模型
        model.eval()
        val_loss = 0.0  # 验证集的总损失
        avg_precision, avg_recall, avg_iou = 0.0, 0.0, 0.0  # 累加每个指标
        val_precision_front, val_recall_front, val_iou_front = 0, 0, 0
        val_precision_back, val_recall_back, val_iou_back = 0, 0, 0

        with torch.no_grad():
            for val_batch_idx, (input_info, ground_truth) in enumerate(val_loader):
                input_info = input_info.cuda()
                ground_truth = ground_truth.cuda()
                # 使用 autocast 进行前向传播
                with torch.amp.autocast('cuda'):
                    output = model(input_info)
                    # loss = criterion(output, ground_truth)  # 计算验证损失
                    # 前半部分的损失 (batch_size, 1, depth//2, height, width)
                    loss_front = criterion(output[:, :, :depth_size // 2, :, :],
                                           ground_truth[:, :, :depth_size // 2, :, :])

                    # 后半部分的损失 (batch_size, 1, depth//2, height, width)
                    loss_back = criterion(output[:, :, depth_size // 2:, :, :],
                                          ground_truth[:, :, depth_size // 2:, :, :])

                # 计算加权后的平均损失
                weighted_loss = 2.0 * loss_front.mean() + loss_back.mean()
                # val_loss += weighted_loss.item()  # 累加验证损失

                # Output shape:  torch.Size([batch_size, 1, 100, 75, 100])
                # Ground truth shape:  torch.Size([batch_size, 1, 100, 75, 100])
                # 计算每个 batch 的指标
                precision, recall, iou = calculate_precision_recall(output, ground_truth)
                # 计算精度、召回率和IoU
                precision_front, recall_front, iou_front = calculate_precision_recall(
                    output[:, :, :depth_size // 2, :, :], ground_truth[:, :, :depth_size // 2, :, :])

                precision_back, recall_back, iou_back = calculate_precision_recall(
                    output[:, :, depth_size // 2:, :, :], ground_truth[:, :, depth_size // 2:, :, :])

                wandb.log({
                    "batch_val_loss": weighted_loss.item(),
                    "batch_val_precision": precision,
                    "batch_val_recall": recall,
                    "batch_val_iou": iou,
                    "batch_val_precision_front": precision_front,
                    "batch_val_recall_front": recall_front,
                    "batch_val_iou_front": iou_front,
                    "batch_val_precision_back": precision_back,
                    "batch_val_recall_back": recall_back,
                    "batch_val_iou_back": iou_back
                })
                print(
                    f"Validation (Batch {val_batch_idx + 1}/{len(val_loader)}) loss: {weighted_loss.item():.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, IoU: {iou:.4f}",
                    flush=True)

                # 累积损失和度量
                val_loss += weighted_loss.item()

                val_precision_front += precision_front
                val_recall_front += recall_front
                val_iou_front += iou_front
                val_precision_back += precision_back
                val_recall_back += recall_back
                val_iou_back += iou_back

                avg_precision += precision
                avg_recall += recall
                avg_iou += iou

        # 计算验证集上的平均指标
        num_batches = len(val_loader)
        avg_val_loss = val_loss / num_batches
        avg_precision /= num_batches
        avg_recall /= num_batches
        avg_iou /= num_batches
        avg_precision_front = val_precision_front / num_batches
        avg_recall_front = val_recall_front / num_batches
        avg_iou_front = val_iou_front / num_batches
        avg_precision_back = val_precision_back / num_batches
        avg_recall_back = val_recall_back / num_batches
        avg_iou_back = val_iou_back / num_batches

        wandb.log({
            "epoch_val_loss": avg_val_loss,
            "epoch_val_precision": avg_precision,
            "epoch_val_recall": avg_recall,
            "epoch_val_iou": avg_iou,
            "epoch_val_precision_front": avg_precision_front,
            "epoch_val_recall_front": avg_recall_front,
            "epoch_val_iou_front": avg_iou_front,
            "epoch_val_precision_back": avg_precision_back,
            "epoch_val_recall_back": avg_recall_back,
            "epoch_val_iou_back": avg_iou_back
        })

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}, Average Precision: {avg_precision:.4f}, Average Recall: {avg_recall:.4f}, Average IoU: {avg_iou:.4f}",
            flush=True)
        print(
            f"Precision (Front): {avg_precision_front:.4f}, Recall (Front): {avg_recall_front:.4f}, IoU (Front): {avg_iou_front:.4f}",
            flush=True)
        print(
            f"Precision (Back): {avg_precision_back:.4f}, Recall (Back): {avg_recall_back:.4f}, IoU (Back): {avg_iou_back:.4f}",
            flush=True)

        # 训练完成后保存模型
        model_path = os.path.join(f'model_checkpoints_1208_bs_16_lr_{config.learning_rate}_pw_{config.positive_weight}',
                                  f'model_epoch_{epoch + 1}.pth')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 如果有其他需要保存的信息，可以在这里添加
        }, model_path)

        print(f'Model saved to {model_path}', flush=True)

    # 保存模型并将其上传到 wandb
    torch.save(model.state_dict(), "voxel_model_1208.pth")
    wandb.save("voxel_model_1208.pth")  # 完成 wandb 运行
    wandb.finish()

    # 完成
    print("Training complete.", flush=True)
