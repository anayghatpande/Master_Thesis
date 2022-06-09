import glob
import time
from multiprocessing import Process

from skimage import io
import numpy as np
import open3d as o3d
import cv2
import json
import logging
import os
from functools import partial
import multiprocessing as mp
from multiprocessing.pool import Pool
from time import time
import shutil
import copy
from pypcd import pypcd
import pprint
import pyvista as pv
from pyvista import examples
from pyntcloud import PyntCloud
import pandas as pd

# from pcd_new_converter import
# device = o3d.core.Device("CPU:0")
# dtype = o3d.core.float32
#dataset_path = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/hope_val/val/*"))

#object_path = sorted(glob.glob("/home/iiwa-2/Downloads/hope_models/models/*ply"))

dataset = input("Choose Dataset or enter path:")
if dataset == str(1):
    dataset_path = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/hope_val/val/*"))
    object_path = sorted(glob.glob("/home/iiwa-2/Downloads/hope_models/models/*ply"))
    output_dataset = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/HOPE_S3ID/open3D/data/s3dis/FLW_dataset/*"))
    dataset_name = "HOPE"
elif dataset == str(2):
    dataset_path = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/ycbv/train_pbr/*"))
    object_path = sorted(glob.glob("/home/iiwa-2/Downloads/ycbv_models/models/*ply"))
    output_dataset = sorted(glob.glob("/media/iiwa-2/MEDIA/YCB_S3DIS/open3D/data/s3dis/FLW_dataset/*"))
    dataset_name = "YCB-V"
else:
    dataset_path = dataset
    output_path = input("Choose Dataset or enter path:")
    object_path_given = input("enter models path:")
    object_path = sorted(glob.glob(os.path.join(object_path_given, '*')))
    output_dataset = sorted(glob.glob(os.path.join(output_path, '*')))
    dataset_name = input("Enter Dataset name")


def json_loader(dataset_no):
    print("Selected Dataset is: ", str(dataset_no))
    for i in range(len(dataset_path)):
        json_path = dataset_path[i] + "/scene_gt.json"
        print(json_path)
        file = open(json_path)
        #shutil.rmtree(dataset_path[i] + "/pcd_annotated", ignore_errors=True)
        #print("Removing old pcd_annotated")
        #os.mkdir(path + "/pcd_annotated")
        dict = json.load(file)
        # print(dict, len(dict['1']))

        # print(dict['0'][0])
        # pcd = o3d.t.geometry.PointCloud(device)
        for scene_id in range(len(dict)):
            global instance_id
            global obj_ids
            instance_id = []
            obj_ids = []
            labels = []
            global cloud_annotation_data
            cloud_annotation_data = list()
            pcd_path = output_dataset[i] + "/pcd/" + str(scene_id) + ".ply"
            # if not os.path.exists(pcd_path):
            #     break
            #     #os.mkdir(txt_path)
            #     #print("Directory ", pcd_img,  " Created ")
            # else:
            #     print("Directory ", pcd_path,  " does not exists... waiting for next")
            #     time.sleep(100)
            #print(pcd_path)
            cloud = PyntCloud.from_file(pcd_path)
            df = pd.DataFrame(cloud.points)
            path_annot = sorted(glob.glob(os.path.join(output_dataset[i], '*')))
            #print(path_annot)
            #os.mkdir(path + "/pcd_annotated/" + str(scene_id))

            # print("scene no. ", scene_id)
            for no_of_objs in range(len(dict[str(scene_id)])):
                # print(objs, dict[str(scene_id)][objs]['obj_id'])
                objects = dict[str(scene_id)][no_of_objs]['obj_id']
                camR = dict[str(scene_id)][no_of_objs]['cam_R_m2c']
                camT = dict[str(scene_id)][no_of_objs]['cam_t_m2c']
                R = np.array(camR)
                shape = (3, 3)
                matR = R.reshape(shape)
                Tr = np.array(camT)
                shape = (3, 1)
                matT = Tr.reshape(shape)
                object_data = transform(matR, matT, objects, pcd_path, path_annot[scene_id])

        print("PCD Annotated", scene_id)


def transform(matr, matt, objts, pcd_path, path_annotation):
    scene = o3d.io.read_point_cloud(pcd_path)  # load scene
    pcd_obj = o3d.io.read_point_cloud(object_path[objts - 1])  # load object

    pcd_obj.points = o3d.utility.Vector3dVector(
        np.array(pcd_obj.points) / 1000)

    # pcd = o3d.io.read_triangle_mesh(m).sample_points_poisson_disk(5000)
    T = np.eye(4)
    # T[:3, :3] = pcd.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
    T[:3, :3] = matr
    T[:3, 3:] = matt / 1000
    pcd_t = copy.deepcopy(pcd_obj).transform(T)
    pcd_t.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    obj_ids.append(objts)
    inst = obj_ids.count(objts)

    pcd_tree = o3d.geometry.KDTreeFlann(scene)  # KDTREE of scene
    seg_points = np.zeros(len(scene.points), dtype=int)

    for point in pcd_t.points:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, 0.005)
        # np.asarray(scene.colors)[idx[1:], :] = [0, 1, 0]
        seg_points[idx[1:]] = objts
        #print(np.asarray(scene.colors))

    seg_idx = np.where(seg_points == objts)[0]
    obj_data = {"type": str(objts),
                "instance": str(instance_id.count(objts)),
                "point_indices": list(map(str, seg_idx))
                }
    #print(seg_points[idx])
    scene2 = PyntCloud.from_file(pcd_path)
    #print(seg_idx)
    color_data= np.asarray(scene.colors)[seg_idx] * 255
    xyz_data= np.asarray(scene.points)[seg_idx]

    df_obj_color = pd.DataFrame(color_data, dtype=int, index=None)
    df_obj_xyz = pd.DataFrame(xyz_data, index=None)
    df_obj = pd.concat([df_obj_xyz, df_obj_color], axis=1)
    #print(df_obj)
    txt_path = path_annotation + "/object" + str(objts-1) + "_" + str(inst) + ".txt"
    if not os.path.exists(txt_path):
        df_obj.to_csv(txt_path, index=False, header=False, sep=' ')
        print("writing sample" + path_annotation[62:] + " obj no." + str(objts-1))

    return objts, seg_points, inst, objts


def spawn_process(number):
    print(f'Runs in separate process {number}')


if __name__ == '__main__':
    max_processes = 30

    print(f'Start {max_processes} processes')

    all_processes = mp.Process(target=json_loader, args=(dataset_name,))
    all_processes.start()
    all_processes.join()

    print('Finished running all Processes')
