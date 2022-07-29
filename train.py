import os
from pathlib import Path

import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d.ml.torch.models import KPFCNN, PointTransformer
from open3d.ml.torch.pipelines import SemanticSegmentation

# from open3d.ml.datasets import FLWDataset
from configuration_file import *

# Reading the Dataset
from torch.testing._internal.common_utils import args

framework = "torch"  # or torch

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# KPCONV MODEL
#HOPE CONFIG
#cfg_file = "/home/iiwa-2/Frameworks/Open3D-ML/ml3d/configs/kpvconv_FLW_HOPE.yml"
cfg_file = dataset_config[4]
#YCB CONFIG
#cfg_file = "/home/iiwa-2/Frameworks/Open3D-ML/ml3d/configs/kpconv_FLW_YCB.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

#dataset_name = 'FLWDATASETS3DIS'
#dataset_name = 'FLWDATASETHOPE'
dataset_name = dataset_config[3]

# model = ml3d.models.KPFCNN(**cfg.model)
#YCB
#dataset_path = "/media/iiwa-2/MEDIA/YCB_S3DIS/open3D/data/s3dis/FLW_data"
#HOPE
#dataset_path = "/home/iiwa-2/Downloads/Datasets/HOPE_S3ID/open3D/data/s3dis/s3dis_data"
dataset_path = dataset_config[5]
if dataset_name == "FLWDATASETHOPE":
    print("Selected Dataset: ", dataset_name)
    dataset = ml3d.datasets.FLWDATASETHOPE(dataset_path=dataset_path, name=dataset_name)
elif dataset_name == "FLWDATASETS3DIS":
    print("Selected Dataset: ", dataset_name)
    dataset = ml3d.datasets.FLWDATASETS3DIS(dataset_path=dataset_path, name=dataset_name)
else:
    print("adding new dataset please create proper file for it")
    print("Selected Dataset: ", dataset_name)
    dataset = ml3d.datasets.Custom3D(dataset_path=dataset_path, name=dataset_name)

# dataset_path_S3DIS ="/home/iiwa-2/Downloads/Stanford3dDataset_v1.2"

#test_split = dataset.get_split('validation')
# dataset_S3DIS =ml3d.datasets.S3DIS(dataset_path=dataset_path_S3DIS)
# dataset = ml3d.datasets.FLWDATASETS3DIS(dataset_path=dataset_path,
#                                       cache_dir='./logs/cache',
#                                       training_split=['00'],
#                                       validation_split=['01'],
#                                       test_split=['01'])

# train_split = dataset.get_split('training')
# train_split = dataset_S3DIS.get_split('training')
# training own model

# use a cache for storing the results of the preprocessing (default path is './logs/cache')

# create the model with initialization.

model = KPFCNN(**cfg.model)

#
pipeline = SemanticSegmentation(model, dataset=dataset, device="gpu", num_workers=0, pin_memory=False, **cfg.pipeline)
# pipeline = SemanticSegmentation(model=model,
#                                 dataset=dataset,
#                                 max_epoch=100,
#                                 optimizer={'lr': 0.001},
#                                 num_workers=0)

# # prints training progress in the console.
pipeline.run_train()

# Testing data
pipeline.run_test()

