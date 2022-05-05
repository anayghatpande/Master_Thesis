import glob
from skimage import io
import numpy as np
import open3d as o3d
import cv2
import json
import os
import shutil
from pyntcloud import PyntCloud
dataset = input("Choose Dataset or enter path:")
if dataset == str(1):
    dataset_path = "/home/iiwa-2/Downloads/Datasets/hope_val/val/"
    model_path = sorted(glob.glob("/home/iiwa-2/Downloads/hope_models/models/*ply"))
    print("Selected Dataset is: HOPE")
elif dataset == str(2):
    dataset_path = "/home/iiwa-2/Downloads/Datasets/ycbv/train_pbr/"
    model_path = sorted(glob.glob("/home/iiwa-2/Downloads/ycbv_models/models/*ply"))
    print("Selected Dataset is: YCB-V")
else:
    dataset_path = dataset
    model_path = input("enter models path:")


#dataset_path = "/home/iiwa-2/Downloads/hope_val/val/000001"
dataset_read = sorted(glob.glob(dataset_path + "/*"))
print("Removing old POINT-CLOUDS...")


def load_json(data_path):
    file = open(data_path + "000001/scene_camera.json")
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
    rgb_path = sorted(glob.glob(r + "/rgb/*"))
    return rgb_path


def depth_reader(d):
    # reading all depth images
    depth_path = sorted(glob.glob(d + "/depth/*"))
    return depth_path


def pcd_writer(path):
    # removing all old pcd images
    shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)
    return path

def bin_remover(path):
    # removing all old pcd images
    shutil.rmtree(path, ignore_errors=True)
    #os.mkdir(path)
    return path

dataset = list(dataset_read)
config = load_json(dataset_path)


#@jit(nopython=True)
def converter():
    for i in range(len(dataset)):
        rgb_images = rgb_reader(dataset[i])
        depth_images = depth_reader(dataset[i])
        pcd_path = str(dataset[i]) + "/pcd"
        bin_path = str(dataset[i]) + "/bin"
        pcd_img = pcd_writer(pcd_path)
        bin_files = bin_remover(bin_path)

        for j in range(len(depth_images)):
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

            # downpcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=0.05)
            # o3d.geometry.PointCloud.estimate_normals(
            #     downpcd,
            #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
            #                                                       max_nn=30))
            o3d.io.write_point_cloud(str(pcd_img) + "/" + str(j) + ".ply", pcd)
            #scene = o3d.io.read_point_cloud(str(pcd_img) + "/" + str(j) + ".ply")
            #o3d.visualization.draw_geometries([scene]) # debug

            # alpha = 0.03
            # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            # mesh.compute_vertex_normals()
            # o3d.io.write_point_cloud(str(pcd_img) + "/" + str(j) + ".ply", mesh)
            # my_point_cloud = PyntCloud.from_file(str(pcd_img) + "/" + str(j) + ".pcd")
            #
            # my_point_cloud.to_file(str(bin_files) + "/" + str(j) + ".bin")
            # #pcd2 = o3d.io.read_point_cloud(str(bin_files) + "/" + str(j) + ".txt", format='xyzrgb')
            # my_point_cloud2 = PyntCloud.from_file(str(bin_files) + "/" + str(j) + ".bin")
            #print(my_point_cloud2) #debug
            #open_cloud = PyntCloud.from_file(str(bin_files) + "/" + str(j) + ".bin")
            #print(open_cloud)


        print("writing point-clouds in scene", i)
    #        o3d.visualization.draw_geometries([pcd]) #debug


    print("Conversion Successful")



if __name__ == "__main__":
    converter()
