from PIL import Image
import pandas as pd
import numpy as np
import open3d as o3d
from open3d import read_point_cloud, draw_geometries
import time
print("Load a ply point cloud, print it, and render it")
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

print("Testing IO for point cloud ...")
sample_pcd_data = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
print(pcd)
o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)
