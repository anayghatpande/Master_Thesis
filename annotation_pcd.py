import glob
from skimage import io
import numpy as np
import open3d as o3d
import cv2
import json
import os
import shutil
import copy
from pypcd import pypcd
import pprint
import pyvista as pv
from pyvista import examples
from pyntcloud import PyntCloud
import pandas as pd

#from pcd_new_converter import
device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32
dataset_path = sorted(glob.glob("/home/iiwa-2/Downloads/Datasets/hope_val/val/*"))

object_path = sorted(glob.glob("/home/iiwa-2/Downloads/hope_models/models/*ply"))


def json_loader():

    for path in dataset_path:
        json_path = path + "/scene_gt.json"
        print(json_path)
        file = open(json_path)
        shutil.rmtree(path + "/pcd_annotated", ignore_errors=True)
        print("Removing old pcd_annotated")
        os.mkdir(path + "/pcd_annotated")
        dict = json.load(file)
        #print(dict, len(dict['1']))


        #print(dict['0'][0])
        pcd = o3d.t.geometry.PointCloud(device)
        for scene_id in range(len(dict)):
            global instance_id
            global obj_ids
            instance_id =[]
            obj_ids = []
            labels = []
            global cloud_annotation_data
            cloud_annotation_data = list()
            pcd_path = path+"/pcd/" + str(scene_id) + ".ply"
            cloud = PyntCloud.from_file(pcd_path)
            df = pd.DataFrame(cloud.points)
            #print("scene no. ", scene_id)
            for no_of_objs in range(len(dict[str(scene_id)])):
                #print(objs, dict[str(scene_id)][objs]['obj_id'])
                objects = dict[str(scene_id)][no_of_objs]['obj_id']
                camR = dict[str(scene_id)][no_of_objs]['cam_R_m2c']
                camT = dict[str(scene_id)][no_of_objs]['cam_t_m2c']
                R = np.array(camR)
                shape = (3, 3)
                matR = R.reshape(shape)
                Tr = np.array(camT)
                shape = (3, 1)
                matT = Tr.reshape(shape)
                object_data = transform(matR, matT, objects, pcd_path)
                #point_cloud = pv.PolyData(np.asarray(object_data[2].points))
                #data = object_data[1]
                labels.append("object_" + str(object_data[0]))
                df[labels[no_of_objs]] = object_data[1]
                #df2 = pd.DataFrame(object_data[1], columns=list("object_" + str(object_data[0])))
                #df.append(df2)



            labelsfile = path + "/labels/0"+str(scene_id)+".label"
            os.makedirs(os.path.dirname(labelsfile), exist_ok=True)
            with open(labelsfile, "w") as f:
                f.write(str(labels))
                f.close()
            #print(labels[no_of_objs])



                #object_ids = cloud.add_scalar_field("obj_id", object_id=object_data[1])
                #object_id = cloud.add_structure("obj_id", object_data[1:])
                #print(cloud.points)


            print("PCD Annotated")
            #print(df)




            cloud_new = PyntCloud(df)

            #o3d.visualization.draw_geometries([cloud_new])
            cloud_new.to_file(path + "/pcd_annotated/"+str(scene_id)+".ply")
            os.makedirs(os.path.dirname(path + "/npz/"), exist_ok=True)
            cloud_new.to_file(path + "/npz/"+str(scene_id)+".npz")

            # cloud2 = PyntCloud.from_file(path + "/pcd_annotated/"+str(scene_id)+".bin")
            # df2 = pd.DataFrame(cloud2.points)
            # print(df2)
            #converted_pc = cloud2.to_instance("open3d", mesh=False)
            #test_cloud = o3d.io.read_point_cloud(path + "/pcd_annotated/"+str(scene_id)+".ply")
            #o3d.visualization.draw_geometries([converted_pc])


                # new_cloud = cloud.get_sample("object_ids", object_id=object_id, as_PyntCloud=True)
                # new_cloud.to_file(path + "/pcd_annotated"+scene_id+".pcd")


                #
                # #print(point_cloud, data)
                # point_cloud["obj_"+str(object_data[0])] = data
                #print(len(data))
                #print(object_data[0], instance_id.count(object_data[0]))
                #point_cloud.plot()
            # print(scene_id, labels)

            #file = open(dataset_path)
                #o3d.visualization.draw_geometries([object_data[2]])


            # obj_ids = np.asarray(obj_ids)
            # n = obj_ids.size
            # #N = np.asarray(scene_data.points).size
            # shape = (1, 1890103)
            # object_f = obj_ids.reshape(shape)
            #
            # print(object_f)



                #point_indices = list(map(str, scene_data[1]))
            #print(obj_ids, instance_id)
            #point_cloud = pv.PolyData(np.asarray(scene_data.points))
           # print(point_cloud)
            #
           # print(obj_ids)

            # data = obj_ids
            # point_cloud["elevation"] = data
            #np.allclose(np.asarray(scene_data.points), point_cloud.points)


            # pcd.point["obj_ids"] = o3d.core.Tensor(obj_ids, o3d.core.int32, device)
            # scene_final = o3d.t.geometry.PointCloud.append(scene_data, pcd)


            #print(str(no_of_objs) +" "+ str(objects) +" "+ str(camR) +" " + str(camT))
            #print(no_of_objs, object_path[objects-1])


            # for obj_nos in range(len(dict[str(scene_id)])):
            #     print(obj_nos, dict[str(scene_id)][objs]['obj_id'])

    #
    # for scene_id in range(len(dict)):
    #     dict1 = dict[str(scene_id)]
    #
    #
    # print(dict1)

        # for d in range(len(dictionary)):
        #     print(d)
        #     scene_objects.append(dictionary[d])



def transform(matr, matt, objts, pcd_path):
    #print(path + "/pcd/" + str(scn_id) + ".pcd", objts)

    scene = o3d.io.read_point_cloud(pcd_path)  # load scene
    pcd_obj = o3d.io.read_point_cloud(object_path[objts-1])  # load object



    #pcd_T.append(pcd_m)
    #pcd = o3d.io.read_triangle_mesh(model_path).sample_points_poisson_disk(5000)


    #with open(open(json_cloud_annotation_path, 'w') as f:
    #pcd_obj.paint_uniform_color([0, 0, 1])  # debug

    pcd_obj.points = o3d.utility.Vector3dVector(
            np.array(pcd_obj.points) / 1000)

    #pcd = o3d.io.read_triangle_mesh(m).sample_points_poisson_disk(5000)
    T = np.eye(4)
    # T[:3, :3] = pcd.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
    T[:3, :3] = matr
    T[:3, 3:] = matt/1000

    # print(T[:3, 3:])
    # print(T)

    pcd_t = copy.deepcopy(pcd_obj).transform(T)
    pcd_t.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #print('Displaying original and transformed geometries ...')
    #o3d.visualization.draw_geometries([scene, pcd_t]) # debug
    #
    # obj_ids.append(objts)
    # instance_id.append(objts)
    pcd_tree = o3d.geometry.KDTreeFlann(scene)  # KDTREE of scene
    seg_points = np.zeros(len(scene.points), dtype=bool)

    for point in pcd_t.points:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, 0.005)
        #np.asarray(scene.colors)[idx[1:], :] = [0, 1, 0]
        seg_points[idx[1:]] = True

    seg_idx = np.where(seg_points == True)[0]
    obj_data = {"type": str(objts),
                "instance": str(instance_id.count(objts)),
                "point_indices": list(map(str, seg_idx))
                }


    return objts, seg_points

    #print(obj_data["point_indices"], len(obj_data["point_indices"]))
    #print(seg_points, len(seg_points))
    #print(pcd_tree.set_matrix_data(seg_points))





           # print(obj_ids)



    #print(point_cloud)

    #o3d.io.write_point_cloud("/home/iiwa-2/Downloads/demo_pcd" + "/" + str(scn_id) + ".pcd", point_cloud)



    #o3d.visualization.draw_geometries([scene])
    # obj_data = {"obj_id": str(objts),
    #             "instance": str(instance_id.count(objts)),
    #             "point_indices": list(map(str, seg_idx))
    #             }


    # cloud_annotation_data.append(obj_data)


    # cloud = pypcd.PointCloud.from_path(pcd_path)
    # pprint.pprint(cloud.get_metadata()) # debug
    # print(cloud.pc_data['width'])

    #print(cloud.pc_data.view(np.float32).reshape(cloud.pc_data.shape + (-1,)))
    #cloud = PyntCloud.from_file(pcd_path)
    #pcl = o3d.t.geometry.PointCloud(device)
    #scene.point["obj_ids"] = o3d.core.Tensor(obj_ids, o3d.core.int32, device)
   # o3d.io.write_point_cloud(str(path + "/pcd_annotated") + "/" + str(j) + ".pcd", pcd)
    #print()




    #print(obj_data)




    #o3d.visualization.draw_geometries([pcd_m, pcd_t]) # debug






def annotator():
    pcd_m = o3d.io.read_point_cloud(dataset_path[0] + "/pcd/0.pcd")
    pcd = o3d.io.read_point_cloud(object_path[15])
    pcd.points = o3d.utility.Vector3dVector(
            np.array(pcd.points) / 1000)
    T = np.eye(4)
    #T[:3, :3] =


def cloud_loader(path):
    cloud_path = path + "/pcd/" + str(0) + ".pcd"
    cloud = pypcd.PointCloud.from_path(cloud_path)
    pprint.pprint(cloud.get_metadata())
    print(cloud.pc_data[:])


#cloud_loader(dataset_path[0])
json_loader()
