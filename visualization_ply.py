### OPEN3D+PLY ###
import open3d as o3d

# visualizations_for_pw_0.64_origin_lr_0.0008_bs_16_epoch_28/a5af701c-1825-40ce-a9d7-0b7ed08fb43b_99_combined_visualization.png
# pcd = o3d.io.read_point_cloud("predicted_ply_pw_0.64_origin_lr_0.0008_bs_16_epoch_28/a5af701c-1825-40ce-a9d7-0b7ed08fb43b_93_predicted.ply")

# 点设密之后的gt
# pcd = o3d.io.read_point_cloud("knn_test/a5af701c-1825-40ce-a9d7-0b7ed08fb43b_81_gt.ply")

# 之前的预测
# visualizations_for_pw_0.64_origin_lr_0.0008_bs_16_epoch_28/a7a7529d-7799-4213-9ed2-384be3540fe1_126_combined_visualization.png

# group1
# pcd = o3d.io.read_point_cloud("input_ply_1208/0a761819-05d1-4647-889b-a726747201b1_12_input_info.ply")

pcd = o3d.io.read_point_cloud("single_redicted_ply/0a761819-05d1-4647-889b-a726747201b1-copy_1207_predicted.ply")
# group2
# pcd = o3d.io.read_point_cloud("single_redicted_ply/a5af701c-1825-40ce-a9d7-0b7ed08fb43b_82_predicted.ply")

# group3

# query points
# a7a7529d-7799-4213-9ed2-384be3540fe1_126
# pcd = o3d.io.read_point_cloud("knn_test/a7a7529d-7799-4213-9ed2-384be3540fe1_126_query_points_roi.ply")

# 可视化点云
o3d.visualization.draw_geometries([pcd])


