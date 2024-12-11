import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import json
import time

import matplotlib
matplotlib.use('Agg')  # 设置为无头模式


def render_ply_default(ply_path, save_path):
    # 加载PLY文件
    point_cloud = o3d.io.read_point_cloud(ply_path)

    # 放大物体（例如放大两倍）
    scale_factor = 1.0
    point_cloud.scale(scale_factor, center=point_cloud.get_center())

    # 创建一个可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 将点云添加到窗口
    vis.add_geometry(point_cloud)

    # 渲染并更新视图
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()

    # 截图并保存（直接从默认视图方向）
    vis.capture_screen_image(save_path)

    # 关闭可视化窗口
    vis.destroy_window()
    print(f"Screenshot saved to {save_path}")

def render_ply(ply_path, save_path=None):
    # 加载PLY文件
    point_cloud = o3d.io.read_point_cloud(ply_path)

    # 放大物体（例如放大两倍）
    scale_factor = 1.0
    point_cloud.scale(scale_factor, center=point_cloud.get_center())

    # 创建一个可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 将点云添加到窗口
    vis.add_geometry(point_cloud)

    # 渲染并更新视图
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()

    # 设置视图参数
    view_control = vis.get_view_control()

    # 设置相机位置（在物体的斜上方）
    camera_position = np.array([0.0, 0.0, 1.0])  # 根据物体的实际位置调整此位置
    view_control.set_lookat(camera_position)  # 注视相机位置

    # 设置相机注视的点（物体的中心）
    lookat_point = np.array([0.0, 0.0, 0.0])  # 物体中心位置
    view_control.set_lookat(lookat_point)

    # 设置相机的前方方向（从相机位置指向注视点）
    front_vector = lookat_point - camera_position  # 从相机位置指向注视点
    front_vector /= np.linalg.norm(front_vector)  # 归一化
    view_control.set_front(front_vector)

    # 设置相机的上方向（通常为Y轴正方向）
    up_vector = np.array([0.0, 1.0, -1.0])  
    view_control.set_up(up_vector)

    # 更新视图
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()

    # 截图并保存
    vis.capture_screen_image(f"{save_path}")

    # 关闭可视化窗口
    vis.destroy_window()

def load_metrics(metrics_folder, identifier, last_part):
    """从metrics文件夹中加载精确度、召回率和IoU值"""
    metrics_file = f"{metrics_folder}/{identifier}_{last_part}_metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return metrics.get("precision", 0), metrics.get("recall", 0), metrics.get("iou", 0)
    return None, None, None

def visualize_ply(metrics_folder, predicted_folder, gth_input_folder, save_dir):
    """
    遍历预测输出和 ground truth，生成三张并排的图片并保存。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 遍历预测文件
    cnt = 0
    for pred_file in os.listdir(predicted_folder):
        if pred_file.endswith(".ply"):
            base_name = os.path.splitext(os.path.basename(pred_file))[0]
            
            identifier = base_name.split('_')[0]
            print(f"Predicting for {identifier}")
            last_part = base_name.split('_')[1]
            print(f"last_part: {last_part}")
            
            # 找到相应的 input 和 ground truth 文件
            input_path = os.path.join(gth_input_folder, f"{identifier}_{last_part}_input_info.ply")
            ground_truth_path = os.path.join(gth_input_folder, f"{identifier}_{last_part}_ground_truth.ply")
            output_path = os.path.join(predicted_folder, pred_file)
            
            if os.path.exists(input_path) and os.path.exists(ground_truth_path):
                # 设置保存路径
                img_paths = {
                    "input": os.path.join(save_dir, f"{identifier}_{last_part}_input_info_screenshot.png"),
                    "ground_truth": os.path.join(save_dir, f"{identifier}_{last_part}_ground_truth_screenshot.png"),
                    "output": os.path.join(save_dir, f"{identifier}_{last_part}_output_screenshot.png")
                }
                
                # 渲染并保存各个视图
                render_ply_default(input_path, save_path=img_paths["input"])
                time.sleep(0.1)  # 等待一下
                render_ply_default(ground_truth_path, save_path=img_paths["ground_truth"])
                time.sleep(0.1)  # 等待一下
                render_ply_default(output_path, save_path=img_paths["output"])

                # 加载模型指标
                precision, recall, iou = load_metrics(metrics_folder, identifier, last_part)
                
                # 合成并排图像
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                titles = ["Input", "Ground Truth", "Output"]

                for i, (key, title) in enumerate(zip(img_paths.keys(), titles)):
                    img = plt.imread(img_paths[key])
                    axes[i].imshow(img)
                    axes[i].axis("off")
                    axes[i].set_title(title)

                # 在合成图像的上方添加指标文本
                fig.suptitle(f'{identifier}_{last_part}_visualization.png\nPrecision: {precision:.2f}, Recall: {recall:.2f}, IoU: {iou:.2f}', fontsize=12, y=0.85)

                # 保存三图合成后的图片
                combined_img_path = os.path.join(save_dir, f"{identifier}_{last_part}_combined_visualization.png")
                plt.savefig(combined_img_path, bbox_inches='tight')
                plt.close(fig)

                # 删除原来的三张图片
                for img_path in img_paths.values():
                    if os.path.exists(img_path):
                        os.remove(img_path)
                
                print(f"Visualization saved at {combined_img_path}")
                cnt += 1
                if cnt == 60:
                    break

# 改
metrics_folder = "model_metrics_pw_0.64_origin_lr_0.0008_bs_16_epoch_28"
# 改
predicted_folder = "predicted_ply_pw_0.64_origin_lr_0.0008_bs_16_epoch_28"
gth_input_folder = "test_dataset_gt_input_ply_1111"
# 改
save_dir = "visualizations_for_pw_0.64_origin_lr_0.0008_bs_16_epoch_28"
visualize_ply(metrics_folder, predicted_folder, gth_input_folder, save_dir)

# import open3d as o3d
# import numpy as np
# from PIL import Image
# import os

# def render_ply_gif(ply_path, save_path, num_frames=36, rotation_axis="z"):
#     # os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     # 加载PLY文件
#     point_cloud = o3d.io.read_point_cloud(ply_path)
    
#     # 放大物体
#     scale_factor = 1.0
#     point_cloud.scale(scale_factor, center=point_cloud.get_center())

#     # 创建可视化窗口
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(visible=False)  # 隐藏窗口，加快渲染

#     # 将点云添加到窗口
#     vis.add_geometry(point_cloud)
#     vis.update_geometry(point_cloud)
    
#     # 获取视图控制器
#     view_control = vis.get_view_control()
#     images = []
    
#     # 扫描一圈并保存每个角度的截图
#     for i in range(num_frames):
#         # 旋转视角
#         angle = 360.0 / num_frames
#         if rotation_axis == "y":
#             view_control.rotate(angle, 0.0)
#         elif rotation_axis == "x":
#             view_control.rotate(0.0, angle)
#         elif rotation_axis == "z":
#             view_control.rotate(angle, angle)  # 自定义旋转轴
#         else:
#             raise ValueError("Invalid rotation axis. Choose from 'x', 'y', or 'z'.")
        
#         # 渲染更新
#         vis.poll_events()
#         vis.update_renderer()

#         # 保存截图到内存
#         img_path = f"temp_frame_{i}.png"
#         vis.capture_screen_image(img_path)
#         images.append(Image.open(img_path))

#     # 关闭窗口
#     vis.destroy_window()

#     # 将所有图片合成为 GIF
#     images[0].save(save_path, save_all=True, append_images=images[1:], duration=100, loop=0)

#     # 删除临时文件
#     for img in images:
#         img.close()
#     for i in range(num_frames):
#         os.remove(f"temp_frame_{i}.png")

#     print(f"GIF saved to {save_path}")

# # 调用函数，生成旋转的 GIF
# render_ply_gif("test_dataset_gt_input_ply_1111/a5af701c-1825-40ce-a9d7-0b7ed08fb43b_7_ground_truth.ply", "rotating_point_cloud.gif", num_frames=36, rotation_axis="y")



# def combine_images(img_paths, titles, save_path):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     for i, (path, title) in enumerate(zip(img_paths, titles)):
#         img = plt.imread(path)
#         axes[i].imshow(img)
#         axes[i].axis("off")
#         axes[i].set_title(title)
    
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Combined visualization saved at {save_path}")



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