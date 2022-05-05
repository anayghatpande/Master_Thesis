from skimage import io
import numpy as np
import open3d as o3d
import cv2
import json
import os
import shutil
import copy
import glob

dataset_path = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/hope_val/val/*"))
print(dataset_path[0])
model_path = sorted(glob.glob("/home/iiwa-2/Downloads/hope_models/models/*ply"))
sample_path = "/home/iiwa-2/Downloads/hope_models/models/obj_000005.ply"
#print(len(model_path))
main_scene_0 = "/home/iiwa-2/Downloads/Datasets/hope_val/val/000001/pcd/0.pcd"

def load_data(id):
    camR = []
    camT = []
    obj = []
    file = open(dataset_path[id] + "/scene_gt.json")
    data = json.load(file)
    #print(data['0'], len(data))
    for k in range(len(data['0'])):
        cam = data['0'][k]
        camR.append(cam['cam_R_m2c'])
        camT.append(cam['cam_t_m2c'])
        obj.append(cam['obj_id'])
    #print(dataset_path[0] + "/scene_camera.json")

    R = np.array(camR[0])
    shape = (3, 3)
    matR = R.reshape(shape)
    Tr = np.array(camT[0])
    shape = (3, 1)
    matT = Tr.reshape(shape)
    print(matT)
    print(matR)
    return obj, matR, matT

def select_models(obj_ids):
    #print(obj_ids)
    current_model_path = []
    for objt in obj_ids:
        print(model_path[objt-1])
        current_model_path.append(model_path[objt-1])

    return current_model_path


def transform(matr, matt, model):
    i = 0
    #pcd_T = []
    # armadillo_data = o3d.data.ArmadilloMesh()
    pcd_m = o3d.io.read_point_cloud(main_scene_0)
    #pcd_T.append(pcd_m)
    #pcd = o3d.io.read_triangle_mesh(model_path).sample_points_poisson_disk(5000)
    for m in model:
        pcd = o3d.io.read_point_cloud(m)
        pcd.points = o3d.utility.Vector3dVector(
            np.array(pcd.points) / 1000)
        #pcd = o3d.io.read_triangle_mesh(m).sample_points_poisson_disk(5000)
        T = np.eye(4)
        # T[:3, :3] = pcd.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
        T[:3, :3] = matr
        T[:3, 3:] = matt/1000

        # print(T[:3, 3:])
        # print(T)

        pcd_t = copy.deepcopy(pcd).transform(T)

        #pcd_T.append(pcd_t)
        #pcd_t.scale(0.5, center=pcd_t.get_center())
        #o3d.io.write_point_cloud(pcd_path + "/" + str(i) + ".ply", pcd_t)
        #i = i + 1

        print('Displaying original and transformed geometries ...')
        o3d.visualization.draw_geometries([pcd_m, pcd_t])
        # o3d.visualization.draw([{
        #     "name": "Original Geometry",
        #     "geometry": pcd_m
        # }, {
        #     "name": "Transformed Geometry",
        #     "geometry": pcd_t
        # }, ],
        #     show_ui=True)


    #o3d.visualization.draw_geometries([pcd_m])
    #print(pcd_T)
dataloader = []
dataloader =load_data(0)
# pcd_path = str(dataset_path[0]) + "/pcd_transformed"
# os.mkdir(pcd_path)
print("loading_object_ids", dataloader[0])
curr_model_path = select_models(dataloader[0])
transform(dataloader[1], dataloader[2], curr_model_path)

#testing KDTREE
# print("Testing kdtree in Open3D...")
# print("Load a point cloud and paint it gray.")
# pcd = o3d.io.read_point_cloud("/home/iiwa-2/Downloads/Datasets/hope_val/val/000001/pcd/0.pcd")
# pcd.paint_uniform_color([0.5, 0.5, 0.5])
# pcd_tree = o3d.geometry.KDTreeFlann(pcd)
# print("Paint the 1500th point red.")
# pcd.colors[1500] = [1, 0, 0]
# print("Find its 200 nearest neighbors, and paint them blue.")
# [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
# np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
# print("Find its neighbors with distance less than 0.2, and paint them green.")
# [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2)
# np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
# print("Visualize the point cloud.")
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.5599,
#                                   front=[-0.4958, 0.8229, 0.2773],
#                                   lookat=[2.1126, 1.0163, -1.8543],
#                                   up=[0.1007, -0.2626, 0.9596])



#helper visualiser




#print(curr_model_path)
# global cam
# cam = []
# def read_json(datasetpath):
#
#     for i in range(len(datasetpath)):
#         file = open(datasetpath[i]+"/scene_gt.json")
#         data = json.load(file)
#         #print(i, data)
#         cam.append(np.array(data))
#         print(cam)




#if __name__ == "__main__":
    #read_json(dataset_path)

    #selection_of_model(json_path)

