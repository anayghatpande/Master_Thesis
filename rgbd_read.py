import open3d
pcd = open3d.io.read_point_cloud("/home/iiwa-2/open3d_data/extract/DemoICPPointClouds/cloud_bin_0.pcd")
print(pcd)
#from open3d.open3d.geometry import create_rgbd_image_from_color_and_depth
#pcd = o3d.io.read_point_cloud("/home/iiwa-2/Downloads/sample_640×426.pcd")
#print(pcd)
#help(o3d)
print("Read Redwood dataset")
redwood_rgbd = open3d.data.SampleRedwoodRGBDImages()
color_raw = open3d.io.read_image(redwood_rgbd.color_paths[0])
depth_raw = open3d.io.read_image(redwood_rgbd.depth_paths[0])
rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)
print(rgbd_image)

plt.subplot(1, 2, 1)
plt.title('Redwood grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Redwood depth image')
plt.imshow(rgbd_image.depth)
plt.show()

# print("Read YCBV dataset")
# color_raw = open3d.io.read_image("/home/iiwa-2/Downloads/rgbd_lobby/lobby/image/000000.jpg")
# depth_raw = open3d.io.read_image("/home/iiwa-2/Downloads/rgbd_lobby/lobby/depth/00000.png")
# rgbd_image1 = open3d.geometry.RGBDImage.create_from_color_and_depth(
#     color_raw, depth_raw)
# print(rgbd_image1)

