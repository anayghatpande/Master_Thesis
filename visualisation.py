import os
from os import path
import logging
from os.path import exists, join
import pickle
import numpy as np
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
import glob

import pandas as pd
from open3d.ml.torch.vis import Visualizer, LabelLUT
from open3d.ml.torch.models import KPFCNN, PointTransformer
from open3d.ml.torch.pipelines import SemanticSegmentation
#from open3d.ml.examples import util

example_dir = os.path.dirname(os.path.realpath(__file__))

log = logging.getLogger(__name__)

# Reading the Dataset

framework = "torch"  # or tf
#construct a dataset by specifying dataset_path
dataset = ml3d.datasets.FLWDATASETS3DIS(dataset_path='/media/iiwa-2/MEDIA/YCB_S3DIS/open3D/data/s3dis/FLW_data')

cfg_file = "/home/iiwa-2/Frameworks/Open3D-ML/ml3d/configs/kpconv_FLW_YCB.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)
#
# # get the 'all' split that combines training, validation and test set
# all_split = dataset.get_split('all')
#
# # print the attributes of the first datum
# print(all_split.get_attr(0))
#
# # print the shape of the first point cloud
# print(all_split.get_data(0)['point'].shape)
#
# # show the first 100 frames using the visualizer
# vis = ml3d.vis.Visualizer()
# vis.visualize_dataset(dataset, 'all', indices=range(1))
#


#
# # see this function in examples/vis_pred.py,
# # or it can be your customized dataloader,
# # or you can use the existing get_data() methods in ml3d/datasets

# with open('/media/iiwa-2/MEDIA/YCB_S3DIS/open3D/data/s3dis/FLW_data/original_pkl/s3dis_FLW_dataset_0_10.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(pd.DataFrame(data[0]))
#     print(pd.DataFrame(data[1]))


def get_custom_data(pc_names, path):

    pc_data = []
    # for i, name in enumerate(pc_names):
    #     pc_path = join(path, name + '_point.npy')
    #     label_path = join(path, name + '_ins_label.npy')
    #     point = np.load(pc_path)[:, 0:3]
    #     feat = np.load(pc_path)[:, 3:6]
    #     label = np.squeeze(np.load(label_path))
    #     #print(label[14389])
    #
    #     data = {
    #         'name': name,
    #         'point': point,
    #         'feat': feat,
    #         'label': label,
    #     }
    #     pc_data.append(data)
    #
    # return pc_data
    train_split = dataset.get_split("train")
    pc_data0 = train_split.get_data(0)
    pc_data.append(pc_data0)
    pc_data1 = train_split.get_data(1)
    pc_data.append(pc_data1)
    return pc_data


def pred_custom_data(pc_names, pcs, pipeline_k):
    vis_points = []
    for i, data in enumerate(pcs):
        name = pc_names[i]
        print(data)
        results_k = pipeline_k.run_inference(data)
        pred_label_k = (results_k['predict_labels'] + 1).astype(np.int32)
        # Fill "unlabeled" value because predictions have no 0 values.
        pred_label_k[0] = 0

        label = data['label']
        pts = data['point']
        print(pts)

        vis_d = {
            "name": name,
            "points": pts,
            "labels": label,
            "pred": pred_label_k,
        }
        vis_points.append(vis_d)

        vis_d = {
            "name": name + "_kpconv",
            "points": pts,
            "labels": pred_label_k,
        }
        vis_points.append(vis_d)

    return vis_points





#print(pcs)
# vis = ml3d.vis.Visualizer()
# vis.visualize(pcs)
#Visualizer.visualize(pcs)

#
#
# import open3d.ml.torch as ml3d
# # or import open3d.ml.tf as ml3d
# import numpy as np
#
# num_points = 100000
# points = np.random.rand(num_points, 3).astype(np.float32)
#
# data = [
#     {
#         'name': 'my_point_cloud',
#         'points': points,
#         'random_colors': np.random.rand(*points.shape).astype(np.float32),
#         'int_attr': (points[:,0]*5).astype(np.int32),
#     }
# ]
#
# vis = ml3d.vis.Visualizer()
# vis.visualize(data)


def main():
    FLW_labels = ml3d.datasets.FLWDATASETS3DIS.get_label_to_names()
    v = Visualizer()
    lut = LabelLUT()
    for val in sorted(FLW_labels.keys()):
        lut.add_label(FLW_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)

    #ckpt_path = "./logs_02_07/KPFCNN_FLWDATASETS3DIS_torch/checkpoint/ckpt_00100.pth"
    #ckpt_path = "/home/iiwa-2/Frameworks/Master_Thesis/logs/KPFCNN_FLWDATASETS3DIS_torch/checkpoint/ckpt_00030.pth"
    #print(ckpt_path)
    model = ml3d.models.KPFCNN(**cfg.model)
    pipeline_k = SemanticSegmentation(model, dataset=dataset, device="cpu", num_workers=0, pin_memory=False,  **cfg.pipeline)
    pipeline_k.load_ckpt(model.cfg.ckpt_path)

    data_path = "/media/iiwa-2/MEDIA/YCB_S3DIS/open3D/data/s3dis/FLW_data/"  # from examples/util.py, downloads demo data
    pc_names = ["open3D_FLW_dataset_0_20", "open3D_FLW_dataset_0_30"]
    pcs = get_custom_data(pc_names, data_path)
    #print(pcs)
    pcs_with_pred = pred_custom_data(pc_names, pcs, pipeline_k)
    v.visualize(pcs_with_pred)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )

    main()
