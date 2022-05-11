import os
import glob
import shutil

dataset_path = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/hope_val/val/*"))
dataset_name = "HOPE"
# dataset = input("enter dataset path:")
# dataset_path = sorted(glob.glob(dataset + "/*"))
# dataset_name = input("enter dataset name:")

dirName = "/home/iiwa-2/Downloads/Datasets/" + dataset_name

def structure_maker():
    bin_files_path = []
    for i in range(len(dataset_path)):
        dirName = "/home/iiwa-2/Downloads/Datasets/" + dataset_name + "/dataset/sequences/0" + str(i)
        shutil.rmtree(dirName, ignore_errors=True)
        os.makedirs(dirName)
        bin_files_path.append(sorted(glob.glob(dataset_path[i] + "/bin/*")))
        #os.makedirs(dirName + "/pcd_files")
        #os.makedirs(dirName + "velodyne") # debug
        os.makedirs(dirName + "/labels")
        src_dir = dataset_path[i] + "/bin"
        dest_dir = dirName + "/bin"
        print(src_dir, dest_dir)
        files = os.listdir(src_dir)
        shutil.copytree(src_dir, dest_dir)

    return bin_files_path, dirName

#bin_path = structure_maker()


def create_train_directory():
    shutil.rmtree(dirName, ignore_errors=True)
    os.makedirs(os.path.dirname(dirName + "/train"), exist_ok=True)
    j = 0

    for i in dataset_path:
        src_dir = i + "/pcd"
        dest_dir = dirName + "/train/0" + str(j) + "/pcd/"
        # j = j + 1
        #print(src_dir, dest_dir)
        files = os.listdir(src_dir)
        #print(files)
        shutil.copytree(src_dir, dest_dir)
    # for k in dataset_path:
        src_dir2 = i + "/npy"
        dest_dir2 = dirName + "/train/0" + str(j) + "/npy/"
        j = j + 1
        #print(src_dir2, dest_dir2)
        files2 = os.listdir(src_dir2)
        #print(files2)
        shutil.copytree(src_dir2, dest_dir2)


#print(bin_path)
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
create_train_directory()










