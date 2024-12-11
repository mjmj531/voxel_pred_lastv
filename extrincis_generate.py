# 1210 把x轴下的旋转角改对了,改了一下角度范围

import bpy
import numpy as np
import mathutils
import math

def generate_random_camera_extrinsics():
    # 随机生成相机位置
    # 是否要改为正态分布？
    x = np.random.uniform(-2, 2)
    y = np.random.uniform(1.2, 1.3)
    z = np.random.uniform(-3, 3)
    # print("随机生成的相机位置：", x, y, z)
    camera_location = mathutils.Vector((x, y, z))

    # 随机生成旋转角度
    rot_x = np.random.uniform(-20, 10) # 默认左闭右开
    rot_y = np.random.uniform(-180, 180)
    rot_z = np.random.uniform(-20, 20)
    # print("随机生成的相机旋转角度：", rot_x, rot_y, rot_z)
    rot_x_rad = math.radians(rot_x)  # 绕x轴 [-10°, 10°]
    rot_y_rad = math.radians(rot_y)  # 绕y轴 [-180°, 180°]
    rot_z_rad = math.radians(rot_z)  # 绕z轴 [-20°, 20°]

    # 创建相机的欧拉角
    camera_rotation = mathutils.Euler((rot_x_rad, rot_y_rad, rot_z_rad), 'XYZ')

    # 取出各个轴的旋转角度
    rotation_x, rotation_y, rotation_z = camera_rotation.x, camera_rotation.y, camera_rotation.z

    # 将绕y轴的旋转角取反
    inverted_rotation_y = -rotation_y

    # 检查绕y轴的旋转角的范围，并决定是否取反x和z轴的角度
    if -math.pi <= inverted_rotation_y < 0:  # 对应 -180° 到 0°
        inverted_rotation_x = -rotation_x
        inverted_rotation_z = -rotation_z
    else:  # 对应 0° 到 180°
        inverted_rotation_x = -rotation_x
        inverted_rotation_z = rotation_z

    # 创建新的欧拉角
    inverted_rotation = mathutils.Euler((inverted_rotation_x, inverted_rotation_y, inverted_rotation_z), 'XYZ')

    # 根据取反的欧拉角生成旋转矩阵
    inverted_rotation_matrix = inverted_rotation.to_matrix().to_4x4()

    # 生成位移矩阵 (世界坐标位置)
    translation_matrix = mathutils.Matrix.Translation(camera_location)

    # 新的外参矩阵 = 位移矩阵 * 取反后的旋转矩阵
    new_camera_extrinsics = translation_matrix @ inverted_rotation_matrix

    # 将外参矩阵转换为 Python 的可序列化形式 (4x4 列表)
    new_camera_extrinsics_list = [[new_camera_extrinsics[i][j] for j in range(4)] for i in range(4)]

    return new_camera_extrinsics_list