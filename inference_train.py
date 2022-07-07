#Training Semantic Segmentation Model using PyTorch
import os
#Import torch and the model to use for training
import open3d.ml.torch as ml3d
import open3d.ml as _ml3d
from open3d.ml.torch.vis import Visualizer, LabelLUT
from open3d.ml.torch.models import KPFCNN, PointTransformer
from open3d.ml.torch.pipelines import SemanticSegmentation
import pandas as pd
cfg_file = "/home/iiwa-2/Frameworks/Open3D-ML/ml3d/configs/kpconv_FLW_YCB.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

#Create a checkpoint
model = ml3d.models.KPFCNN(**cfg.model)
dataset = ml3d.datasets.FLWDATASETS3DIS(dataset_path='/media/iiwa-2/MEDIA/YCB_S3DIS/open3D/data/s3dis/FLW_data', use_cache=False)
#KPFCNN = Model(ckpt_path=args.KPFCNN)
pipeline = SemanticSegmentation(model, dataset=dataset, device="cpu", num_workers=0, pin_memory=False,  **cfg.pipeline)

#Get data from the SemanticKITTI dataset using the "train" split
train_split = dataset.get_split("train")
data = train_split.get_data(0)
data1 = train_split.get_data(1)

print(len(data1))
#Run the inference
results = pipeline.run_inference(data)

#Print the results
print(pd.DataFrame(results['predict_labels']))
print(pd.DataFrame(results['predict_scores']))
#vis = ml3d.vis.Visualizer()
#vis.visualize(data)
