import numpy
import numpy as np
import glob
import open3d as o3d
import cv2
import json
import os
file = open("/home/iiwa-2/Downloads/Datasets/ycbv/train_pbr/000000/scene_camera.json")
data = json.load(file)
print(data['0']['cam_K'])
mat = data['0']['cam_K']
file.close()
intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, mat[0], mat[4], mat[2], mat[5])


def read_instances(path, iteration):
    if iteration < 10:
        path = path + "00" + str(iteration)
    elif iteration < 100:
        path = path + "0" + str(iteration)
    else:
        path = path + str(iteration)
    return path


dataset_path = "/home/iiwa-2/Downloads/Datasets/ycbv/train_pbr/000"

# pcd_dir = dataset_path + "/000"+str(i)+"/pcd"
# os.system('rm -r /home/iiwa-2/Downloads/Datasets/ycbv/train_pbr/000000x/pcd')
# print(pcd_path)
# os.mkdir(pcd_dir)
#pcd_path = glob.glob("/home/iiwa-2/Downloads/Datasets/ycbv/train_pbr/000000x/depth/*pcd")
#f = 0


def read_image(path, iteration, imagetype):
    if iteration < 10:
        path = path + "00" + str(iteration) + str(imagetype)
    elif iteration < 100:
        path = path + "0" + str(iteration) + str(imagetype)
    else:
        path = path + str(iteration) + str(imagetype)
    return path


def write_pcd_image(path, iteration):
    if iteration < 10:
        path = path + "00" + str(iteration) + str(".pcd")
    elif iteration < 100:
        path = path + "0" + str(iteration) + str(".pcd")
    else:
        path = path + str(iteration) + str(".pcd")
    return path


# x = read_instances(dataset_path, 47)
# print(x)


for i in range(0, 50):
    rgb_path = str(read_instances(dataset_path, i)) + "/rgb/000"
    depth_path = str(read_instances(dataset_path, i)) + "/depth/000"
    pcd_dir = str(read_instances(dataset_path, i)) + "/pcd"
    os.mkdir(pcd_dir)
    # print(pcd_dir)
    pcd_path = str(read_instances(dataset_path, i)) + "/pcd/000"
    # print(rgb_path)
    # print(pcd_path)
    for f in range(0, 1000):

        color_raw = cv2.imread(read_image(rgb_path, f, ".jpg"))
        img = numpy.array(color_raw)
        color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)
        color_raw = o3d.geometry.Image(color_raw)
        # cv2.imshow("Image", img)
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows()

        depth_raw = cv2.imread(read_image(depth_path, f, ".png"), -1)/1000000
        depth_raw = np.float32(depth_raw)
        depth_raw = o3d.geometry.Image(depth_raw)
    #
    #
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=0.1, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        o3d.io.write_point_cloud(write_pcd_image(pcd_path, f), pcd)

#    o3d.visualization.draw_geometries([pcd])


#


