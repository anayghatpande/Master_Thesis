import numpy
import numpy as np

import open3d as o3d
import cv2
import json
import os
file = open("/home/iiwa-2/Downloads/Datasets/ycbv/train_pbr/000000/scene_camera.json")
data = json.load(file)
print(data['0']['cam_K'])
mat = data['0']['cam_K']
# for a in range(0,5):
#     data1 = data[str(a)]['cam_K']
#     print(data1)


file.close()
#mat = [1390.53, 0.0, 964.957, 0.0, 1386.99, 522.586, 0.0, 0.0, 1.0]
intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, mat[0], mat[4], mat[2], mat[5])

color_raw = cv2.imread("/home/iiwa-2/Downloads/Datasets/ycbv/train_pbr/000000/rgb/000000.jpg")
color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)
color_raw = o3d.geometry.Image(color_raw)

depth_raw = cv2.imread("/home/iiwa-2/Downloads/Datasets/ycbv/train_pbr/000000/depth/000000.png", -1)/1000000
depth_raw = np.float32(depth_raw)
depth_raw = o3d.geometry.Image(depth_raw)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=0.1,
                                                                convert_rgb_to_intensity=False)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
o3d.visualization.draw_geometries([pcd])


