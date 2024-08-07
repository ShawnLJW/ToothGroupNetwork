import open3d as o3d
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from safetensors import safe_open
import json
from external_libs.pointops.functions import pointops
import trimesh

def np_to_pcd(arr, color=[1,0,0]):
    arr = np.array(arr)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    if arr.shape[1] >= 6:
        pcd.normals = o3d.utility.Vector3dVector(arr[:,3:6])
    pcd.colors = o3d.utility.Vector3dVector([color]*len(pcd.points))
    return pcd

def save_pcd(path, arr):
    o3d.io.write_point_cloud(path, arr)

def save_mesh(path, mesh):
    o3d.io.write_triangle_mesh(path, mesh)

def count_unique_by_row(a):
    weight = 1j*np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
    b = a + weight[:, np.newaxis]
    u, ind, cnt = np.unique(b, return_index=True, return_counts=True)
    b = np.zeros_like(a)
    np.put(b, ind, cnt)
    return b

def get_colored_mesh(mesh, labels):
    palte = np.array([
        [230,230,222],

        [153,76,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],

        [255,128,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],
    ])/255
    palte[9:] *= 0.4
    label_colors = labels.copy()
    label_colors[label_colors>30] -= 20
    label_colors[label_colors//10==1] %= 10
    label_colors[label_colors//10==2] = (label_colors[label_colors//10==2]%10) + 8
    label_colors[label_colors<0] = 0
    label_colors = palte[label_colors]
    mesh.vertex_colors = o3d.utility.Vector3dVector(label_colors)
    return mesh

def np_to_pcd_with_label(arr, label_arr=None, axis=3):
    if type(label_arr) == np.ndarray:
        arr = np.concatenate([arr[:,:3], label_arr.reshape(-1,1)],axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    
    palte = np.array([
        [255,153,153],

        [153,76,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],

        [255,128,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],
    ])/255
    palte[9:] *= 0.4
    arr = arr.copy()
    arr[:,axis] %= palte.shape[0]
    label_colors = np.zeros((arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[arr[:,axis]==idx] = palte[idx]
    pcd.colors = o3d.utility.Vector3dVector(label_colors)
    return pcd

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def get_number_from_name(path):
    return int(os.path.basename(path).split("_")[-1].split(".")[0])

def get_up_from_name(path):
    return os.path.basename(path).split("_")[-1].split(".")[0]=="up"

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    if pcd.shape[0] == n:
        return pcd
    elif pcd.shape[0] > n:
        idx = fps(pcd[:,:3], n)
    elif pcd.shape[0] < n:
        idx = torch.randint(0, pcd.shape[0], (n,))
    return pcd[idx] 

def fps(xyz, npoint):
    if xyz.shape[0]<=npoint:
        raise "new fps error"
    xyz = torch.from_numpy(np.array(xyz)).type(torch.float).cuda()
    idx = pointops.furthestsampling(xyz, torch.tensor([xyz.shape[0]]).cuda().type(torch.int), torch.tensor([npoint]).cuda().type(torch.int)) 
    return torch_to_numpy(idx).reshape(-1)

def print_3d(*data_3d_ls, wireframe=False, back_face=True):
    data_3d_ls = [item for item in data_3d_ls]
    for idx, item in enumerate(data_3d_ls):
        if type(item) == np.ndarray:
            data_3d_ls[idx] = np_to_pcd(item)
    o3d.visualization.draw_geometries(
        data_3d_ls,
        mesh_show_wireframe = wireframe,
        mesh_show_back_face = back_face,
    )

def torch_to_numpy(cuda_arr):
    return cuda_arr.cpu().detach().numpy()

def save_np(arr, path):
    with open(path, 'wb') as f:
        np.save(f, arr)

def load_np(path):
    with open(path, 'rb') as f:
        arr = np.load(f)
    return arr

def axis_rotation(axis, angle):
    ang = np.radians(angle) 
    R=np.zeros((3,3))
    ux, uy, uz = axis
    cos = np.cos
    sin = np.sin
    R[0][0] = cos(ang)+ux*ux*(1-cos(ang))
    R[0][1] = ux*uy*(1-cos(ang)) - uz*sin(ang)
    R[0][2] = ux*uz*(1-cos(ang)) + uy*sin(ang)
    R[1][0] = uy*ux*(1-cos(ang)) + uz*sin(ang)
    R[1][1] = cos(ang) + uy*uy*(1-cos(ang))
    R[1][2] = uy*uz*(1-cos(ang))-ux*sin(ang)
    R[2][0] = uz*ux*(1-cos(ang))-uy*sin(ang)
    R[2][1] = uz*uy*(1-cos(ang))+ux*sin(ang)
    R[2][2] = cos(ang) + uz*uz*(1-cos(ang))
    return R

def make_coord_frame(size=1):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])

def load_json(file_path):
    with open(file_path, "r") as st_json:
        return json.load(st_json)

def save_json(file_path, json_obj):
    with open(file_path, "w") as json_file:

        json.dump(json_obj, json_file)

def load_labels(file_path):
    with open(file_path, "r") as f:
        if file_path.endswith(".txt"):
            gt = [int(x) for x in f.read().split("\n") if x != ""]
            gt = np.array(gt)
            gt[gt > 8] = 7 - gt[gt > 8]
            gt[gt != 0] = 19 - gt[gt != 0]
        elif file_path.endswith(".json"):
            gt = json.load(f)["labels"]
            gt = np.array(gt)
            
    return gt

def read_txt(file_path):
    f = open(file_path, 'r')
    path_ls = []
    while True:
        line = f.readline().split()
        if not line: break
        path_ls.append(os.path.join(os.path.dirname(file_path), line.split("\n")[0] + ".npy"))
    f.close()

    return path_ls

def load_mesh(path):
    tri_mesh_loaded_mesh = trimesh.load_mesh(path, process=True)
    vertex_ls = np.array(tri_mesh_loaded_mesh.vertices)
    tri_ls = np.array(tri_mesh_loaded_mesh.faces)+1
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls)-1)
    mesh.compute_vertex_normals()

    norms = np.array(mesh.vertex_normals)
    feats = np.concatenate([vertex_ls,norms], axis=1)

    return feats, mesh

def load_checkpoint(path):
    if path.endswith(".pt") or path.endswith(".pth") or path.endswith(".h5"):
        return torch.load(path)
    elif path.endswith(".safetensors"):
        tensors = {}
        with safe_open(path, "pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors
    else:
        raise ValueError("Unknown file format")