# reading dataset path
import glob
import os


def create_open3d_dataset(dataset_BOP, dataset_o3d, dataset_name):
    os.makedirs(os.path.join(os.path.join(os.path.join(dataset_o3d, dataset_name), 'data'), 'meta_data'))
    open3d_path = os.path.join(os.path.join(os.path.join(os.path.join(dataset_o3d, dataset_name), 'data'), 'FLW_dataset'))
    os.makedirs(open3d_path)
    for d in range(len(dataset_BOP)):
        samples = dataset_BOP[d].split('/')
        os.makedirs(os.path.join(open3d_path, samples[-1]))

    return open3d_path

def set_ur_dataset():
    dataset = input("Choose Dataset from HOPE(1) or YCB(2) or DoPose(3) or Custom path:")
    if dataset == str(1):
        dataset_BOP = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/hope_val/val/*"))
        model_path = sorted(glob.glob("/home/iiwa-2/Downloads/hope_models/models/*ply"))
        dataset_open3d = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/HOPE_S3ID/open3D/data/s3dis/FLW_dataset/*"), key=os.path.basename)
        dataset_name = "FLWDATASETHOPE"
        cfg_file = "/home/iiwa-2/Frameworks/Open3D-ML/ml3d/configs/kpvconv_FLW_HOPE.yml"
        training_data = "/home/iiwa-2/Downloads/Datasets/HOPE_S3ID/open3D/data/s3dis/FLW_data"

    elif dataset == str(2):
        dataset_BOP = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/ycbv/train_pbr/*"))
        model_path = sorted(glob.glob("/home/iiwa-2/Downloads/ycbv_models/models/*ply"))
        dataset_open3d = sorted(glob.glob("/media/iiwa-2/MEDIA/YCB_S3DIS/open3D/data/s3dis/FLW_dataset/*"), key=len)
        #print(*output_dataset, sep='\n')
        dataset_name = "FLWDATASETS3DIS"
        cfg_file = "/home/iiwa-2/Frameworks/Open3D-ML/ml3d/configs/kpconv_FLW_YCB.yml"
        training_data = "/media/iiwa-2/MEDIA/YCB_S3DIS/open3D/data/s3dis/FLW_data"
    elif dataset == str(3):
        dataset_BOP = sorted(glob.glob("/media/iiwa-2/Datasets/DoPose/test_bin/*"))
        model_path = sorted(glob.glob("/media/iiwa-2/Datasets/DoPose/models/models/*ply"))
        dataset_open3d = sorted(glob.glob("/media/iiwa-2/Datasets/DoPose_Train/DoPose_bin/data/FLW_dataset/*"))
        dataset_name = "DoPose"
        cfg_file = "nothing"
        training_data = "nothing"

    else:
        dataset_BOP = sorted(glob.glob(os.path.join(dataset, '*')))
        model_path = sorted(glob.glob(os.path.join(input("enter models path:"), '*')))
        dataset_o3d = input("Define open3d dataset path: ")
        dataset_name = input("Enter Dataset name: ")
        dataset_open3d = sorted(glob.glob(os.path.join(create_open3d_dataset(dataset_BOP, dataset_o3d, dataset_name), '*')))
        cfg_file = "nothing"
        training_data = "nothing"
    return dataset_BOP, model_path, dataset_open3d, dataset_name, cfg_file, training_data


dataset_config = set_ur_dataset()

# print(dataset_config[0])
# print(dataset_config[1])
# print(dataset_config[2])
# print(dataset_config[3])

