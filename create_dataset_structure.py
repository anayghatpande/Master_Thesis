import os
import glob
import shutil

dataset_path = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/hope_val/val/*"))
dataset_name = "HOPE"
# main_paths = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/HOPE_S3ID/mmdet3d/data/s3dis/FLW_dataset/*"))
# # for i in range(len(main_paths)):
# #     x = os.path.join(main_paths[i], 'Annotations/')
# #     #file = open('/home/iiwa-2/Frameworks/mmdetection3d_anay/data/s3dis/meta_data/anno_paths.txt', 'w')
# #     for j in sorted(glob.glob(x +'/*')):
# #         #print(j)
#
#
#     #anno_paths = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/HOPE_S3ID/mmdet3d/data/s3dis/FLW_dataset/0"+str(i) +"/Annotations/*"))
#     #print(x)
#     # with open('/home/iiwa-2/Frameworks/mmdetection3d_anay/data/s3dis/meta_data/anno_paths.txt', 'w') as file:
#     #     file.writelines(anno_paths)
#
#     #
#     #     file.write(anno_paths + '\n')
#
# dataset = input("enter dataset path:")
# dataset_path = sorted(glob.glob(dataset + "/*"))
# dataset_name = input("enter dataset name:")

#dirName2 = "/home/iiwa-2/Downloads/Datasets/" + dataset_name

def structure_maker():
    txt_files_path = []
    for i in range(len(dataset_path)):
        #dirName = "/home/iiwa-2/Downloads/Datasets/" + dataset_name + "/dataset/sequences/0" + str(i)
        dirName = "/home/iiwa-2/Downloads/Datasets/HOPE_S3ID/mmdet3d/data/s3dis/FLW_dataset/Scene0" + str(i)
        shutil.rmtree(dirName, ignore_errors=True)
        #os.makedirs(dirName)
        txt_files_path.append(sorted(glob.glob(dataset_path[i] + "/npy/*")))

        #os.makedirs(dirName + "/pcd_files")
        #os.makedirs(dirName + "velodyne") # debug
       # for j in range(len(txt_files_path)):
        #print(len(txt_files_path[0]))
        for j in range(len(txt_files_path[0])):

            src_dir = dataset_path[i] + "/npy/"
            files = os.listdir(src_dir)
            dest_dir = dirName + "/" + str(j)
            print(src_dir, dest_dir)

            print(files)

        #os.makedirs(dirName + "/Annotations/")
            shutil.copytree(src_dir, dest_dir)
            src_dir2 = dataset_path[i] + "/pcd_annotated/"+ str(j)
            files2 = os.listdir(src_dir2)
            #print(len(files2))
            dest_dir2 = dirName + "/" + str(j) + "/Annotations"
            #print(src_dir2, dest_dir2)

        #print(files2)

            shutil.copytree(src_dir2, dest_dir2)


    # return txt_files_path, dirName


structure_maker()


def create_train_directory():
    shutil.rmtree(dirName2, ignore_errors=True)
    os.makedirs(os.path.dirname(dirName2 + "/train"), exist_ok=True)
    j = 0

    for i in dataset_path:
        src_dir = i + "/pcd"
        dest_dir = dirName2 + "/train/0" + str(j) + "/pcd/"
        # j = j + 1
        #print(src_dir, dest_dir)
        files = os.listdir(src_dir)
        #print(files)
        shutil.copytree(src_dir, dest_dir)
    # for k in dataset_path:
        src_dir2 = i + "/npy"
        dest_dir2 = dirName2 + "/train/0" + str(j) + "/npy/"
        j = j + 1
        #print(src_dir2, dest_dir2)
        files2 = os.listdir(src_dir2)
        #print(files2)
        shutil.copytree(src_dir2, dest_dir2)


#
# # path to source directory
# src_dir = 'fol1'
#
# # path to destination directory
# dest_dir = 'fol2'
#
# # getting all the files in the source directory
# files = os.listdir(src_dir)
#
# shutil.copytree(src_dir, dest_dir)
#create_train_directory(dirName2)

#dir = "/home/iiwa-2/Downloads/Datasets/"
#dataset_name = "HOPE_KITTI"
def structure_maker_KITTI():
    bin_files_path = []
    for i in range(len(dataset_path)):
        dirName = dir + dataset_name + "/dataset/sequences/0" + str(i)
        shutil.rmtree(dirName, ignore_errors=True)
        os.makedirs(dirName)
        bin_files_path.append(sorted(glob.glob(dataset_path[i] + "/bin/*")))
        #os.makedirs(dirName + "/pcd_files")
        #os.makedirs(dirName + "velodyne") # debug
       # os.makedirs(dirName + "/labels")
        src_dir = dataset_path[i] + "/bin"
        dest_dir = dirName + "/bin"
        print(src_dir, dest_dir)
        files = os.listdir(src_dir)
        shutil.copytree(src_dir, dest_dir)

    return bin_files_path, dirName


#bin_path = structure_maker_KITTI()


def create_train_directory_KITTI(dirName):
    shutil.rmtree(dirName, ignore_errors=True)
    os.makedirs(os.path.dirname(dirName + "/train"), exist_ok=True)
    j = 0

    for i in dataset_path:
        src_dir = i + "/pcd"
        dest_dir = dirName + "/0" + str(j) + "/bin/"
        # j = j + 1
        #print(src_dir, dest_dir)
        files = os.listdir(src_dir)
        #print(files)
        shutil.copytree(src_dir, dest_dir)
    # for k in dataset_path:
    #     src_dir2 = i + "/npy"
    #     dest_dir2 = dirName + "/train/0" + str(j) + "/npy/"
    #     j = j + 1
    #     #print(src_dir2, dest_dir2)
    #     files2 = os.listdir(src_dir2)
    #     #print(files2)
    #     shutil.copytree(src_dir2, dest_dir2)


#create_train_directory_KITTI(bin_path[1])

