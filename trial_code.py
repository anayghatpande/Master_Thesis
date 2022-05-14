# import copy
# import glob
# from numba import *
# from numpy import dtype
# from skimage import io
# import numpy
# import numpy as np
# import open3d as o3d
# import cv2
# import json
# import os
# import shutil
# import copy
# import trimesh
#
# model_path = sorted(glob.glob("/home/iiwa-2/Downloads/hope_models/models/*"))
#
# #dataset_path = "/home/iiwa-2/Downloads/hope_val/val/")
# dataset_read = sorted(glob.glob("/home/iiwa-2/Downloads/hope_val/val/*"))
# # print(dataset_read)
# # file_list = sorted(glob.glob("/home/iiwa-2/Downloads/hope_val/val/000001/rgb/*png"))
# # rgb_path = sorted(glob.glob(dataset_read + "/rgb/*png"))
# # depth_path = sorted(glob.glob(dataset_read + "/depth/*png"))
# # print(file_list)
# # Once all the paths are retrieved, read the images one by one in a loop.my_list = []for file in file_list:
# # my_list = []
# # for file in file_list:
# #     im = io.imread(file)
# #     my_list.append(im)
# # print(my_list)
# # print(rgb_path, len(rgb_path))
#
# file = open(dataset_path + "000001/scene_camera.json")
# data = json.load(file)
# print("intrinsic camera parameters", data['0']['cam_K'])
# mat = data['0']['cam_K']
# depth_scale = data['0']['depth_scale']
# print("depth_scale", depth_scale)
# file.close()
# intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, mat[0], mat[4], mat[2], mat[5])
# #for dataload in range():
#
# file2 = open(dataset_path + "000001/scene_gt.json")
# data2 = json.load(file2)
# obj = []
# camR = []
# camT = []
# for m in range(len(data2)):
#     for k in range(len(data2[str(m)])):
#         cam = data2[str(m)][k]
#         camR.append(cam['cam_R_m2c'])
#         camT.append(cam['cam_t_m2c'])
#         obj.append(cam['obj_id'])
#
#     print("Object Ids " + str(m), obj)
#     print('\n')
#     print("Translation " + str(m), camT)
#     print('\n')
#     print("Rotation " + str(m), camR)
#
# #deciding obj_id
#
# #def pointcloud_transformation(camR, camT, obj_id):
#
#
#
#
#     #for obj_id in range(m):
#         #print(obj_id)
#         #camR.append(data2[obj_id]['cam_R_m2c'])
#         #print(obj_id, camR)
#     # for obj_id in range(len(obj)):
#     #     print("Rotation", camR[obj_id])
#
# print(camR)
# main_scene_0 = "/home/iiwa-2/Downloads/hope_val/val/000001/pcd/0.ply"
# # models_path = sorted(glob.glob("/home/iiwa-2/Downloads/hope_models/models/*.ply"))
# # #print(models_path)
# #
# # print("Load a ply point cloud, print it, and render it")
# # pcd = o3d.io.read_point_cloud(models_path[15])
# # #print(np.asarray(pcd.points))
# # R = np.array(camR[0])
# # shape = (3, 3)
# # matR = R.reshape(shape)
# # Tr = np.array(camT[0])
# # shape = (3, 1)
# # matT = Tr.reshape(shape)
# # print(matT)
# # print(matR)
#
#
# #pcd_m.estimate_normals()
# #pcd_m_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_m)
# #o3d.visualization.draw_geometries([pcd_m])
# # rotation = o3d.geometry.PointCloud.rotate()
# # print(rotation)
#
# #@jit(nopython=True)
# def pointcloud_mesh():
#     pcd_m = o3d.io.read_point_cloud(main_scene_0)
#     alpha = 0.001
#     #print(f"alpha={alpha:.3f}")
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_m, alpha)
#     mesh.compute_vertex_normals()
#     o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
# # pcd_m = o3d.io.read_point_cloud(main_scene_0)
# # pcd_m.estimate_normals()
# # distances = pcd_m.compute_nearest_neighbor_distance()
# # avg_dist = np.mean(distances)
# # radius = 1.5 * avg_dist
# # mesh_m = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
# #     pcd_m,
# #     o3d.utility.DoubleVector([radius, radius * 2]))
# #
# # # create the triangular mesh with the vertices and faces from open3d
# # tri_mesh = trimesh.Trimesh(np.asarray(mesh_m.vertices), np.asarray(mesh_m.triangles),
# #                            vertex_normals=np.asarray(mesh_m.vertex_normals))
# #
# # trimesh.convex.is_convex(tri_mesh)
#
#
# def rotate():
#     # armadillo_data = o3d.data.ArmadilloMesh()
#     pcd = o3d.io.read_triangle_mesh(model_path).sample_points_poisson_disk(10000)
#     pcd_r = copy.deepcopy(pcd).translate((200, 0, 0))
#
#     # R = pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
#     # print(R)
#     pcd_r.rotate(matR)
#     print('Displaying original and rotated geometries ...')
#     o3d.visualization.draw([{
#         "name": "Original Geometry",
#         "geometry": pcd
#     }, {
#         "name": "Rotated Geometry",
#         "geometry": pcd_r
#     }, ],
#         show_ui=True)
#
#
# def transform():
#     # armadillo_data = o3d.data.ArmadilloMesh()
#     pcd_m = o3d.io.read_point_cloud(main_scene_0)
#     #pcd = o3d.io.read_triangle_mesh(model_path).sample_points_poisson_disk(5000)
#     pcd = o3d.io.read_point_cloud(model_path)
#
#
#     T = np.eye(4)
#     # T[:3, :3] = pcd.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
#     T[:3, :3] = matR
#     T[:3, 3:] = matT
#
#     # print(T[:3, 3:])
#     # print(T)
#     pcd_t = copy.deepcopy(pcd).transform(T)
#     pcd_t.scale(0.5, center=pcd_t.get_center())
#     print('Displaying original and transformed geometries ...')
#     o3d.visualization.draw([{
#         "name": "Original Geometry",
#         "geometry": pcd
#     }, {
#         "name": "Transformed Geometry",
#         "geometry": pcd_t
#     }, {
#         "name": "main Geometry",
#         "geometry": pcd_m
#     }, ],
#         show_ui=True)
#
#
# # rotate()
# #transform()
# #pointcloud_mesh()
#
# # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# # mesh_tx = copy.deepcopy(mesh).translate((1.3, 0, 0))
# # mesh_ty = copy.deepcopy(mesh).translate((0, 1.3, 0))
# # mesh_tz = copy.deepcopy(mesh).translate((0, 0, 1.3))
# # print(f'Center of mesh: {mesh.get_center()}')
# # print(f'Center of mesh tx: {mesh_tx.get_center()}')
# # print(f'Center of mesh ty: {mesh_ty.get_center()}')
# # print(f'Center of mesh tz: {mesh_tz.get_center()}')
# # o3d.visualization.draw_geometries([mesh, mesh_tx, mesh_ty, mesh_tz])
#
# # def read_any_image(f1):
# #     color_raw = cv2.imread(f1)
# #     #print(f1)
# #     cv2.imshow(str(f1), color_raw)
# #     cv2.waitKey(2000)
# #     cv2.destroyAllWindows()
# #
# #
# # def read_all_images(file_list):
# #     img = cv2.imread(file_list)
# #     #print(file_list)
# #     #cv2.imshow(str(file_list), img)
# #     #cv2.waitKey(2000)
# #     #cv2.destroyAllWindows()
# #     return img
# #
# #
# # # for i in range(0, len(rgb_path)):
# # # for n in dataset_read:
# #     #print(n, '\n')
# #
# #     #print(n)
# #
# # #print(paths, '\n')
# #
# #     # color_raw = rgb_reader(n)
# #     # print(color_raw, '\n')
# #     # depth_raw = depth_reader(n)
# #     # print(depth_raw, '\n')
# #
# # # def converter(rgb_path, depth_path):
# # #     for i in range(0, dataset_read):
# # #         color_raw = cv2.imread(rgb_path[i])
# # #         print(color_raw, i)
# # #         depth_raw = cv2.imread(depth_path[i])
# # #         print(depth_raw, i)
# #
# #
# # def rgb_reader(r):
# #     rgb_path = sorted(glob.glob(r + "/rgb/*png"))
# #     return rgb_path
# #
# #
# # def depth_reader(d):
# #     depth_path = sorted(glob.glob(d + "/depth/*png"))
# #     return depth_path
# #
# #
# # def pcd_writer(path):
# #     shutil.rmtree(path, ignore_errors=True)
# #     os.mkdir(path)
# #     return path
# #
# #
# #
# #
# # data1 = list(dataset_read)
# #
# # #print(len(data))
# # # for i in range(len(data1)):
# # rgb_images = rgb_reader(data1[0])
# # depth_images = depth_reader(data1[0])
# # pcd_path = str(data1[0]) + "/pcd"
# # pcd_img = pcd_writer(pcd_path)
# # #print(pcd_img)
# #
# # #os.mkdir()
# # # for j in range(len(rgb_images)):
# # #print(depth_images[j], i)
# # color_raw = cv2.imread(rgb_images[0])
# # img = numpy.array(color_raw)
# # color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)
# # color_raw = o3d.geometry.Image(color_raw)
# # # cv2.imshow(str(j), img)
# # # cv2.waitKey(2000)
# # # cv2.destroyAllWindows()
# # depth_raw = cv2.imread(depth_images[0], -1)/1000
# # img2 = numpy.array(depth_raw)
# # depth_raw = np.float32(depth_raw)
# # depth_raw = o3d.geometry.Image(depth_raw)
# # # cv2.imshow(str(j), img2)
# # # cv2.waitKey(2000)
# # # cv2.destroyAllWindows()
# #
# # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale, convert_rgb_to_intensity=False)
# # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
# # #o3d.io.write_point_cloud(str(pcd_img) + "/" + str(j) + ".pcd", pcd)
# # #o3d.visualization.draw_geometries([pcd])
# # #print(rgb_reader(data[i]), '\n')
import glob
import gzip
import pickle

import numpy as np
import pandas as pd
import open3d as o3d
from pypcd import pypcd
import pprint
from pyntcloud import PyntCloud


# scene = o3d.io.read_point_cloud("/home/iiwa-2/Downloads/Datasets/hope_val/val/000001/pcd/0.ply")
# scene_converted = o3d.io.read_point_cloud("/home/iiwa-2/Downloads/Datasets/hope_val/val/000001/pcd_annotated/0.ply")
# #df = pd.DataFrame(cloud.points)
# o3d.visualization.draw_geometries([scene, scene_converted])


# cloud = PyntCloud(pd.DataFrame(
#     # same arguments that you are passing to visualize_pcl
#     data=np.hstack((points, colors)),
#     columns=["x", "y", "z", "red", "green", "blue"]))
#print(df)
#cloud.to_file("output.ply")
# dataset_path = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/hope_val/val/*"))
# print(dataset_path[0])
# cloud2 = PyntCloud.from_file(dataset_path[0] + "/pcd_annotated/"+str(0)+".ply")
# df2 = pd.DataFrame(cloud2.points)
# print(df2)
outfile = "/home/iiwa-2/Downloads/Datasets/hope_val/val/000001/npz_annotated/0.npz"
npzfile = np.load(outfile)
#print(npzfile.files)
#print(npzfile['points'])
df = pd.DataFrame(npzfile['points'])
print(df)
#np.savetxt(r'/home/iiwa-2/Downloads/Datasets/hope_val/val/000001/npz_annotated/demo.txt', df)
outfile2 = "/mnt/Media/dataset_zips/Open3D-PointNet2-Semantic3D/dataset/semantic_raw/untermaederbrunnen_station3_xyz_intensity_rgb.labels"
#npzfile2 = np.load()
print(outfile2.shape)
# df2 = pd.read_pickle(outfile2, compression='infer')
# print(df2)


# import open3d.ml.torch as ml3d
# # or import open3d.ml.tf as ml3d
# import numpy as np
#
# num_points = 100000
# points = np.random.rand(num_points, 3).astype(np.float32)
#
# data = [
#     {
#         'name': 'my_point_cloud',
#         'points': points,
#         'random_colors': np.random.rand(*points.shape).astype(np.float32),
#         'int_attr': (points[:,0]*5).astype(np.int32),
#     }
# ]
#
# vis = ml3d.vis.Visualizer()
# vis.visualize(data)
