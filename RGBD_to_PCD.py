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
from open3d.ml.datasets.utils import DataProcessing, get_min_bbox, BEVBox3D
import pandas as pd
import _thread
import threading
import time
import os, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath, isdir
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging

# reading dataset path
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


def load_json(data_path):
    file = open(data_path + "/scene_camera.json")
    data = json.load(file)
    mat = data['0']['cam_K']
    depth_scale = data['0']['depth_scale']
    file.close()
    intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, mat[0], mat[4], mat[2], mat[5])
    if depth_scale < 1:
        # conversion from mm with 0.1 depth scale
        div_factor = float(10 / depth_scale) * 1000
    else:
        div_factor = float(1 / depth_scale) * 1000

    return depth_scale, div_factor, intrinsics


def rgb_reader(r):
    # reading all rgb images
    rgb_path = sorted(glob.glob(r + "/rgb/*"), key=os.path.basename)
    #print(*rgb_path, sep='\n')
    return rgb_path


def depth_reader(d):
    # reading all depth images
    depth_path = sorted(glob.glob(d + "/depth/*"), key=os.path.basename)
    #print(*depth_path, sep='\n')
    return depth_path


def pcd_writer(path):
    # removing all old pcd images
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path

def bin_remover(path):
    # removing all old pcd images
    shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)
    return path

dataset = list(dataset_path)
config = load_json(dataset_path[0])

def load_frm_json(dataset_path, output_dataset, pcd_data):
        json_path = dataset_path + "/scene_gt.json"
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
            objts = []
            # global cloud_annotation_data
            # cloud_annotation_data = list()
            #pcd_path = output_dataset[i] + "/pcd/" + str(scene_id) + ".ply"
            #print(pcd_path)

            #cloud = PyntCloud.from_file(pcd_path)
            #df = pd.DataFrame(cloud.points)
            path_annot = os.path.join(output_dataset, str(scene_id))
            print(path_annot)

            # os.mkdir(path + "/pcd_annotated/" + str(scene_id))

            #print("sample " + str(scene_id) + " of scene " + str(i))
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
                objts.append(objects)
            #print(scene_id, objts)
                #print(path_annot)
                object_data = transformer(matR, matT, objects, path_annot, pcd_data)


#print(depth_images, len(depth_images))
#@jit(nopython=True)
def converter(rgb_images, depth_images):
    pcd_data = []
    for j in range(len(depth_images)):
        #print(rgb_images[j], depth_images[j])
        color_raw = cv2.imread(rgb_images[j])
        img = np.array(color_raw)
        color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)
        color_raw = o3d.geometry.Image(color_raw)
        depth_raw = cv2.imread(depth_images[j], -1) /int(config[1])
        # img2 = np.array(depth_raw)
        depth_raw = np.float32(depth_raw)
        depth_raw = o3d.geometry.Image(depth_raw)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=config[0],
                                                                        convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, config[2])
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd_data.append(pcd)
    return pcd_data


            #print(object_data[2])
            #print(object_data[0], object_data[1])
            #print(pcd)
            #print(dataset[i])



def transformer(matr, matt, objts, path_annotation, pcd_data):
    #print(pcd_path)
    #print(object_path[objts - 1])
    scene = pcd_data  # load scene
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
    #scene2 = PyntCloud.from_file(pcd_data)

    # print(seg_idx)
    color_data = np.asarray(scene.colors)[seg_idx] * 255
    xyz_data = np.asarray(scene.points)[seg_idx]
    #flag = not np.any(color_data) or not np.any(xyz_data)

    df_obj_color = pd.DataFrame(color_data, dtype=int, index=None)
    df_obj_xyz = pd.DataFrame(xyz_data, index=None)
    df_obj = pd.concat([df_obj_xyz, df_obj_color], axis=1)
    #print(df_obj)
    txt_path = path_annotation + "/object" + str(objts - 1) + "_" + str(inst) + ".txt"
    print(txt_path)
    #print("object data", df_obj.shape)

    #print(path_annotation)


    # if not os.path.exists(txt_path):
    #     if df_obj.shape < (50, 6):
    #         print(str(objts - 1), "Object out of instance sample " + path_annotation[62:])
    #     else:
    #         df_obj.to_csv(txt_path, index=False, header=False, sep=' ')
    #     #print("writing sample" + path_annotation[62:] + " obj no." + str(objts - 1))
    # else:
    #     if df_obj.shape < (50, 6):
    #         print("removing old Object which is out of instance")
    #         os.remove(txt_path)

    #o3d.visualization.draw_geometries([scene, pcd_t])  # debug

    return objts, seg_points, inst, objts

# exitFlag = 0
#
# class myThread (threading.Thread):
#    def __init__(self, threadID, name, counter):
#       threading.Thread.__init__(self)
#       self.threadID = threadID
#       self.name = name
#       self.counter = counter
#    def run(self):
#       print("Starting " + self.name)
#       converter()
#       print_time(self.name, 5, self.counter)
#       print("Exiting " + self.name)
#
# def print_time(threadName, counter, delay):
#    while counter:
#       if exitFlag:
#          threadName.exit()
#       time.sleep(delay)
#       print(threadName, time.ctime(time.time()))
#       counter -= 1
#
# # Create new threads
# thread1 = myThread(1, "Thread-1", 1)
# thread2 = myThread(2, "Thread-2", 2)
#
# # Start new Threads
# thread1.start()
# thread2.start()

def create_ply_files(dataset_path, class_names):
        os.makedirs(join(dataset_path, 'original_pkl'), exist_ok=True)
        anno_file = Path(dataset_path) / 'meta_data' / 'anno_paths.txt' # changes acc. to YCB'

        anno_file = str(anno_file)
        anno_paths = [line.rstrip() for line in open(anno_file)]
        anno_paths = [Path(dataset_path) / p for p in anno_paths]

        class_names = [val for key, val in class_names.items()]
        label_to_idx = {l: i for i, l in enumerate(class_names)}

        out_format = '.pkl'

        for anno_path in tqdm(anno_paths):
            elems = str(anno_path).split('/')
            save_path = elems[-4] + '_' + elems[-3] + '_' + elems[-2] + '_' + elems[-1] + out_format
            save_path = Path(dataset_path) / 'original_pkl' / save_path


            data_list = []
            bboxes = []
            for file in glob.glob(str(anno_path / '*.txt')):
                class_name = Path(file).name.split('_')[0]

                #if class_name not in class_names:
                 #   class_name = 'clutter'

                pc = pd.read_csv(file, header=None,
                                 delim_whitespace=True).values
                labels = np.ones((pc.shape[0], 1)) * label_to_idx[class_name]
                data_list.append(np.concatenate([pc, labels], 1))
                bbox = [class_name] + get_min_bbox(pc)
                bboxes.append(bbox)

            pc_label = np.concatenate(data_list, 0)

            info = [pc_label, bboxes]
            if not os.path.exists(save_path):
                with open(save_path, 'wb') as f:
                    pickle.dump(info, f)
            else:
                print("file exists... skipping")




for i in range(len(dataset)):
    #print(len(dataset))
    #shutil.rmtree(output_dataset + str(i), ignore_errors=True)
    rgb_images = rgb_reader(dataset[i])
    depth_images = depth_reader(dataset[i])

    pcd_data = converter(rgb_images, depth_images)
    print(i, len(pcd_data))
    for scn in range(len(pcd_data)):
        print(pcd_data[scn])
        load_data = load_frm_json(dataset[i], output_dataset[i], pcd_data[scn])
    #print(len(object_data[0]), len(object_data[1]))
    #print(object_data[0][1], object_data[1][1])
    #for scn in range(len(x)):
        #print(scn)
        #for t in range(len(object_data[2])):
            #print(len(x[1]))
            #print("OBJECT " + str(object_data[2][t]))
            #print("OBJECT " + str(object_data[2][t]) + "MATR " + str(object_data[0][t]) + "MATT " + str(object_data[1][t]))
        #transformer(object_data[0][t], object_data[1][t], object_data[2][t], object_data[3], x[t])

