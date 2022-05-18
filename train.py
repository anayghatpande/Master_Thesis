import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

cfg_file = "/home/iiwa-2/Frameworks/Open3D-ML/ml3d/configs/kpconv_s3dis.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)
dataset_name = 'HOPE'
model = ml3d.models.KPFCNN(**cfg.model)
cfg.dataset['dataset_path'] = "/home/iiwa-2/Downloads/Datasets/HOPE"
dataset = ml3d.datasets.S3DIS(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
#pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline, split="train")
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname("/home/iiwa-2/Downloads/Datasets/HOPE")
print(ROOT_DIR)
data_dir = os.path.join(ROOT_DIR, dataset_name)
train_data_dir = os.path.join(data_dir, 'train')
print(train_data_dir)
# # download the weights.
# ckpt_folder = "./logs/"
# os.makedirs(ckpt_folder, exist_ok=True)
# ckpt_path = ckpt_folder + "randlanet_semantickitti_202201071330utc.pth"
# randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth"
# if not os.path.exists(ckpt_path):
#     cmd = "wget {} -O {}".format(randlanet_url, ckpt_path)
#     os.system(cmd)
#
# # load the parameters.
# pipeline.load_ckpt(ckpt_path=ckpt_path)
#
# test_split = dataset.get_split("test")
# data = test_split.get_data(0)
#
# # run inference on a single example.
# # returns dict with 'predict_labels' and 'predict_scores'.
# result = pipeline.run_inference(data)
#
# # evaluate performance on the test set; this will write logs to './logs'.
# pipeline.run_test()
