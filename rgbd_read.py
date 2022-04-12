import open3d as o3d
#from open3d.open3d.geometry import create_rgbd_image_from_color_and_depth
#pcd = o3d.io.read_point_cloud("/home/iiwa-2/Downloads/sample_640×426.pcd")
#print(pcd)
#help(o3d)

print("Read YCBV dataset")
color_raw = o3d.io.read_image("/home/iiwa-2/Downloads/rgbd_lobby/lobby/image/000000.jpg")
depth_raw = o3d.io.read_image("/home/iiwa-2/Downloads/rgbd_lobby/lobby/depth/00000.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
print(rgbd_image)

