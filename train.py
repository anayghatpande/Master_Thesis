import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

#Reading the Dataset

# dataset = ml3d.datasets.HopeDataset(dataset_path = "/home/iiwa-2/Downloads/Datasets/HOPE_demo/", name="HopeDataset")
# #dataset = ml3d.datasets.HopeDataset(dataset_path='/home/iiwa-2/Downloads/Datasets/HOPE')
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
# vis.visualize_dataset(dataset, 'all', indices=range(5))

#configuration
from open3d._ml3d.torch import KPFCNN, SemanticSegmentation

framework = "torch" # or tf
cfg_file = "/home/iiwa-2/Frameworks/Open3D-ML/ml3d/configs/kpvconv_HOPE.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)
dataset_name = 'HopeDataset'
model = ml3d.models.KPFCNN(**cfg.model)
dataset_path = "/home/iiwa-2/Downloads/Datasets/HOPE/"
dataset = ml3d.datasets.HopeDataset(dataset_path=dataset_path, name='HopeDataset', train_dir='train', test_dir='test', val_dir='val', use_cache=True)

#dataset = ml3d.datasets.(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline, split="train")
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname("/home/iiwa-2/Downloads/Datasets/HOPE")
# print(ROOT_DIR)
# data_dir = os.path.join(ROOT_DIR, dataset_name)
# train_data_dir = os.path.join(data_dir, 'train')
# print(train_data_dir)

# fetch the classes by the name
Pipeline = _ml3d.utils.get_module("pipeline", cfg.pipeline.name, framework)
Model = _ml3d.utils.get_module("model", cfg.model.name, framework)
Dataset = _ml3d.utils.get_module("dataset", cfg.dataset.name)

# training own model

# use a cache for storing the results of the preprocessing (default path is './logs/cache')
#dataset = ml3d.datasets.HopeDataset(dataset_path='/home/iiwa-2/Downloads/Datasets/HOPE', use_cache=True, name="HopeDataset")

# create the model with initialization.
model = KPFCNN()

pipeline = SemanticSegmentation(model=model, dataset=dataset, max_epoch=100, split='train')

# prints training progress in the console.
pipeline.run_train()

