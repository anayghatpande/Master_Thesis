import glob
import numpy as np
import os
from os import path as osp
import pandas as pd
from configuration_file import *

dataset_path = dataset_config[0]
print(dataset_path)
annotations = sorted(glob.glob(dataset_path), key=os.path.getatime)
anno_dir = dataset_path + "/meta_data/anno_paths.txt"
class_path_dir = dataset_path + "/meta_data/class_names.txt"
model_path = sorted(glob.glob("/home/iiwa-2/Downloads/ycbv_models/models/*.ply"))
#with open(anno_dir, 'w') as file:
#print(annotations)
# df = pd.DataFrame()
# i = 0
# for path in annotations:
#     ind_path = sorted(glob.glob(path + "/*"), key=os.path.getatime)
#     for path_next in ind_path:
#         if path_next == path + "/pcd":
#             print("found pcd path")
#         else:
#             df[str(i)] = ind_path
#     #print(ind_path, "data")
#     print(len(ind_path))
#
#     i = i + 1
# print(df)

    #df = pd.DataFrame(ind_path)


#df = pd.DataFrame()


def data_config():
    if not os.path.exists(anno_dir):
        annotations_path_writer()

    else:
        print("annotation data path file exists ... skipping")

    if not os.path.exists(class_path_dir):
        class_path_txt()
    else:
        print("class names file exists ... skipping")


def annotations_path_writer():
    for path in annotations:
        #print(path)
        ind_path = sorted(glob.glob(path + "/*"), key=os.path.getatime)

        for path_next in ind_path:
            if path_next == path + "/pcd":
                print("found pcd path")
            else:
                print(path_next)
                with open(anno_dir, 'a+') as file:
                    file.write(path_next + "\n")
                    file.close()


def class_path_txt():
    for path in model_path:
        print(path)
        elements = path.split('/')
        class_name = elements[-1].split('.')
        object_no = class_name[-2].split('_')

        with open(class_path_dir, 'a+') as file:
            file.write("object" + str(int(object_no[-1]) - 1) + "\n")
            file.close()
        #print()
        #print(osp.basename(path).split('_'))

# if __name__ == '__main__':
#     data_config()
