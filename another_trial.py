from skimage import io
import numpy as np
import open3d as o3d
import cv2
import json
import os
import shutil
import copy
import glob

dataset_path = sorted(glob.glob("/home/iiwa-2/Downloads/hope_val/val/*"))
#print(dataset_path[0])
model_path = sorted(glob.glob("/home/iiwa-2/Downloads/hope_models/models/*ply"))
sample_path = "/home/iiwa-2/Downloads/hope_models/models/obj_000005.ply"
#print(len(model_path))
main_scene_0 = "/home/iiwa-2/Downloads/hope_val/val/000001/pcd/0.pcd"

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
        pcd = o3d.io.read_triangle_mesh(m).sample_points_poisson_disk(5000)
        T = np.eye(4)
        # T[:3, :3] = pcd.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
        T[:3, :3] = matr
        T[:3, 3:] = matt

        # print(T[:3, 3:])
        # print(T)

        pcd_t = copy.deepcopy(pcd).transform(T)
        #pcd_T.append(pcd_t)
        #pcd_t.scale(0.5, center=pcd_t.get_center())
        #o3d.io.write_point_cloud(pcd_path + "/" + str(i) + ".ply", pcd_t)
        #i = i + 1

        print('Displaying original and transformed geometries ...')
        o3d.visualization.draw_geometries([pcd_m],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
        o3d.visualization.draw([{
            "name": "Original Geometry",
            "geometry": pcd
        }, {
            "name": "Transformed Geometry",
            "geometry": pcd_t
        }, ],
            show_ui=True)


    #o3d.visualization.draw_geometries([pcd_m])
    #print(pcd_T)
dataloader = []
dataloader =load_data(0)
pcd_path = str(dataset_path[0]) + "/pcd_transformed"
#os.mkdir(pcd_path)
#print("loading_object_ids", dataloader[0])
curr_model_path = select_models(dataloader[0])
transform(dataloader[1], dataloader[2], curr_model_path)

#testing KDTREE
print("Testing kdtree in Open3D...")
print("Load a point cloud and paint it gray.")
pcd = o3d.io.read_point_cloud(dataset_path[0] + "/pcd/0.pcd")
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
pcd.colors = [1, 0, 0]

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

