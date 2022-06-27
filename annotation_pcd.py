import datetime
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
from datetime import datetime
import shutil
import copy
from pypcd import pypcd
import pprint
import pyvista as pv
from pyvista import examples
from pyntcloud import PyntCloud
import pandas as pd


dataset = input("Choose Dataset or enter path:")
if dataset == str(1):
    dataset_path = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/hope_val/val/*"))
    object_path = sorted(glob.glob("/home/iiwa-2/Downloads/hope_models/models/*ply"))
    output_dataset = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/HOPE_S3ID/open3D/data/s3dis/FLW_dataset/*"), key=os.path.basename)
    dataset_name = "HOPE"
elif dataset == str(2):
    dataset_path = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/ycbv/train_pbr/*"))
    object_path = sorted(glob.glob("/home/iiwa-2/Downloads/ycbv_models/models/*ply"))
    output_dataset = sorted(glob.glob("/media/iiwa-2/MEDIA/YCB_S3DIS/open3D/data/s3dis/FLW_dataset/*"), key=len)
    #print(*output_dataset, sep='\n')
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
        print(json_path, datetime.now())
        file = open(json_path)
        dict = json.load(file)
        #print(output_dataset[i])

        for scene_id in range(len(dict)):
            global instance_id
            global obj_ids
            instance_id = []
            obj_ids = []
            labels = []
            global cloud_annotation_data
            cloud_annotation_data = list()
            pcd_path = output_dataset[i] + "/pcd/" + str(scene_id) + ".ply"
            #print(pcd_path)

            #cloud = PyntCloud.from_file(pcd_path)
            #df = pd.DataFrame(cloud.points)
            path_annot = os.path.join(output_dataset[i], str(scene_id))
            #print(path_annot)

            # os.mkdir(path + "/pcd_annotated/" + str(scene_id))

            print("sample " + str(scene_id) + " of scene " + str(i))
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
                #print(objects, scene_id)
                #print(path_annot)
                object_data = transform(matR, matT, objects, pcd_path, path_annot)
                # if not os.path.exists(path_annot):
                #     object_data = transform(matR, matT, objects, pcd_path, path_annot)
                # else:
                #     print("path exists skipping for ", scene_id)



        x = output_dataset[i].split('/')
        print("PCD Annotated at " + str(datetime.now()), x[-1])


def transform(matr, matt, objts, pcd_path, path_annotation):
    #print(pcd_path)
    #print(object_path[objts - 1])
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
        # print(np.asarray(scene.colors))

    seg_idx = np.where(seg_points == objts)[0]
    obj_data = {"type": str(objts),
                "instance": str(instance_id.count(objts)),
                "point_indices": list(map(str, seg_idx))
                }
    # print(seg_points[idx])
    scene2 = PyntCloud.from_file(pcd_path)

    # print(seg_idx)
    color_data = np.asarray(scene.colors)[seg_idx] * 255
    xyz_data = np.asarray(scene.points)[seg_idx]
    #flag = not np.any(color_data) or not np.any(xyz_data)

    df_obj_color = pd.DataFrame(color_data, dtype=int, index=None)
    df_obj_xyz = pd.DataFrame(xyz_data, index=None)
    df_obj = pd.concat([df_obj_xyz, df_obj_color], axis=1)
    # print(df_obj)
    txt_path = path_annotation + "/object" + str(objts - 1) + "_" + str(inst) + ".txt"
    #print("object data", df_obj.shape)

    #print(path_annotation)


    if not os.path.exists(txt_path):
        if df_obj.shape < (50, 6):
            print(str(objts - 1), "Object out of instance sample " + path_annotation[62:])
        else:
            df_obj.to_csv(txt_path, index=False, header=False, sep=' ')
        #print("writing sample" + path_annotation[62:] + " obj no." + str(objts - 1))
    else:
        if df_obj.shape < (50, 6):
            print("removing old Object which is out of instance")
            os.remove(txt_path)
        else:
            print("Skipping to next")

    #o3d.visualization.draw_geometries([scene, pcd_t])  # debug

    return objts, seg_points, inst, objts




def spawn_process(number):
    print(f'Runs in separate process {number}')


if __name__ == '__main__':

    all_processes = mp.Process(target=json_loader, args=(dataset_name,))
    all_processes.start()
    all_processes.join()

    print('Finished running all Processes')
