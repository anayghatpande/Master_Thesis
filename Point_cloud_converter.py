import glob
from skimage import io
import numpy as np
import open3d as o3d
import cv2
import json
import os
import shutil
from pyntcloud import PyntCloud
import pandas as pd
from multiprocessing import Process

dataset = input("Choose Dataset or enter path:")
if dataset == str(1):
    dataset_path = "/home/iiwa-2/Downloads/Datasets/hope_val/val/"
    model_path = sorted(glob.glob("/home/iiwa-2/Downloads/hope_models/models/*ply"))
    output_dataset = "/home/iiwa-2/Downloads/Datasets/HOPE_S3ID/open3D/data/s3dis/FLW_dataset/"
    print("Selected Dataset is: HOPE")
elif dataset == str(2):
    dataset_path = "/home/iiwa-2/Downloads/Datasets/ycbv/train_pbr/"
    model_path = sorted(glob.glob("/home/iiwa-2/Downloads/ycbv_models/models/*ply"))
    output_dataset = "/media/iiwa-2/MEDIA/YCB_S3DIS/open3D/data/s3dis/FLW_dataset/"
    print("Selected Dataset is: YCB-V")
else:
    dataset_path = dataset
    output_dataset = input("Choose Dataset or enter path:")
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

dataset = list(dataset_read)
config = load_json(dataset_path)

#print(depth_images, len(depth_images))
#@jit(nopython=True)
def converter():
    for i in range(len(dataset)):
        #shutil.rmtree(output_dataset + str(i), ignore_errors=True)
        rgb_images = rgb_reader(dataset[i])
        depth_images = depth_reader(dataset[i])
        pcd_path = str(output_dataset) + str(i) + "/pcd/"
        npy_path = str(output_dataset) + str(i) + "/"
        #print(pcd_path, npy_path)

        #bin_path = str(dataset[i]) + "/bin"
        #anno_path = str(dataset[i]) + "/annotations"
        pcd_img = pcd_writer(pcd_path)
        #npy_files = pcd_writer(npy_path)
        #bin_files = pcd_writer(bin_path)
        #annotations = pcd_writer(anno_path)

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
            #o3d.visualization.draw_geometries([pcd]) #debug
            txt_path = npy_path + str(j) + "/"
            shutil.rmtree(txt_path, ignore_errors=True)
            #os.remove(txt_path)

            ply_file_path = str(pcd_img) + str(j) + ".ply"
            #print(ply_file_path)
            o3d.io.write_point_cloud(ply_file_path, pcd)
            # if not os.path.exists(ply_file_path):
            #     o3d.io.write_point_cloud(ply_file_path, pcd)
            #     #os.mkdir(txt_path)
            #     #print("Directory ", pcd_img,  " Created ")
            # else:
            #     print("Directory ", ply_file_path,  " already exists... skipping to next")

            if not os.path.exists(txt_path):
                os.mkdir(txt_path)
                #print("Directory ", pcd_img,  " Created ")
            else:
                print("Directory ", txt_path,  " already exists... skipping to next")


            #print(pd.DataFrame(pcd.points))
            #cloud_new = PyntCloud.from_file(str(pcd_img) + str(j) + ".ply")
            #cloud_2 = PyntCloud(pd.DataFrame(cloud_new))
            #array1 = np.asarray(pcd.points)
            #array2 = np.asarray(pcd.colors)
            #print(array2)
            #df = pd.DataFrame(cloud_new.points)
            #print(df[['x', 'y', 'z', 'red', 'green', 'blue']])

            #df.to_csv(txt_path + str(j) + ".txt", index=False, header=False, sep=' ')

            #with open(str(npy_files) + "/" + str(j) + ".txt", 'w') as f:
            #    dfstring = df.to_string(header=False, index=False)
            #    f.write(dfstring)

            #np.savetxt(str(npy_files) + "/" + str(j) + ".txt", df[['x', 'y', 'z', 'red', 'green', 'blue']], fmt='%d')
            #cloud_2.to_file(str(npy_files) + "/" + str(j) + ".npz")
            #o3d.io.write_point_cloud(str(npy_files) + "/" + str(j) + ".txt", pcd)
            #cloud_new.to_file(str(npy_files) + "/" + str(j) + ".txt", cloud_new.points)
                            # sep=" ",
                            # header=0,
                            # points=["x","y","z"])
            #cloud_new.to_file(str(bin_files) + "/" + str(j) + ".bin")
            #
            # #debugging
            # npzfile = np.load(str(npy_files) + "/" + str(j) + ".npz")
            # os.remove(str(npy_files) + "/" + str(j) + ".npz")


            # print(npzfile.files)
            # print(npzfile['points'])
            # with open(str(npy_files) + "/" + str(j) + ".npy", 'wb') as file:
            #     np.save(file, npzfile['points'])
            #     file.close()

            # with open(str(npy_files) + "/" + str(j) + ".txt", 'wb') as file:
            #     np.save(file, npzfile['points'])
            #     file.close()

        print("writing point-clouds in scene", i)

    print("Conversion Successful")

# def info(title):
#     print(title)
#     print('module name:', __name__)
#     print('parent process:', os.getppid())
#     print('process id:', os.getpid())
#
# def f(name):
#     #info('function f')
#     print('hello', name)
#     converter()
#
#
# if __name__ == '__main__':
#     info('main line')
#     p = Process(target=f, args=('process is Running',))
#     p.start()
#     p.join()


if __name__ == "__main__":
    converter()
