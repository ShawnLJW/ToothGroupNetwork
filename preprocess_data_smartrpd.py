import os
import numpy as np
import gen_utils as gu
import open3d as o3d
from sklearn.decomposition import PCA

def read_txt_labels(path):
    with open(path, "r") as f:
        labels = [int(x) for x in f.read().split("\n") if x != ""]
        labels = np.array(labels)
        labels[np.logical_and(labels > 0, labels <= 8)] = (9 - labels[np.logical_and(labels > 0, labels <= 8)])
    
    return labels

folder = "./dataset"
output_dir = "./data_preprocessed_path"

# change the following paths to the correct paths
with open(f"{folder}/test_fold.txt", "r") as f:
    ids = [x for x in f.read().split("\n") if x != ""]

for id in ids:
    jaw = id.split(" ")[1].lower()
    
    jaw_path = f"{folder}/{id}/{jaw}_stage_3.stl"
    label_path = f"{folder}/{id}/{jaw}_stage3results_ordered_v.txt"
    
    if os.path.isfile(jaw_path) and os.path.isfile(label_path):
        feats, mesh = gu.load_mesh(jaw_path)
        vertex_ls = PCA(n_components=3).fit_transform(mesh.vertices)
        vertex_ls[:, 1] *= -1
        mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
        mesh.compute_vertex_normals() 
        labels = read_txt_labels(label_path)
        labeled_vertices = np.concatenate([feats, labels.reshape(-1,1)], axis=1)
        gums_vertices = gu.resample_pcd(labeled_vertices[labels == 0], 12000)
        teeth_vertices = gu.resample_pcd(labeled_vertices[labels != 0], 12000)
        labeled_vertices = np.concatenate([gums_vertices, teeth_vertices], axis=0)
        np.save(f"{output_dir}/{id}_sampled_points", labeled_vertices)