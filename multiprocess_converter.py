import multiprocessing as mp
import os
from multiprocessing import Process, Value
import numpy as np
import numba
from numba import njit
import json
import open3d as o3d
#from Point_cloud_converter import load_json


def load_json(data_path):
    file = open(data_path + "000001/scene_camera.json")
    data = json.load(file)
    mat = data['0']['cam_K']
    depth_scale = data['0']['depth_scale']
    file.close()
    intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, mat[0], mat[4], mat[2], mat[5])
    print("task running")
    os.mkdir('test_dir')

    if depth_scale < 1:
        # conversion from mm with 0.1 depth scale
        div_factor = float(10 / depth_scale) * 1000
    else:
        div_factor = float(1 / depth_scale) * 1000

    return depth_scale, div_factor, intrinsics

results = []
pool = mp.Pool(mp.cpu_count())
print(mp.cpu_count())
dataset_path = "/home/iiwa-2/Downloads/Datasets/hope_val/val/"

result_objects = [pool.apply_async(load_json, dataset_path)]

pool.close()
#p1 = Process(target=load_json(dataset_path))
#p1.start()
#p1.join()
results = [r.get()[1] for r in result_objects]
print(results)

# import dispy
# cluster = dispy.JobCluster('/home/iiwa-2/Frameworks/Master_Thesis/Point_cloud_converter.py')
# for i in range(5):
#     cluster.submit(i)
