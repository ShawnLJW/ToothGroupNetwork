import gen_utils as gu
import numpy as np
import torch
from sklearn.neighbors import KDTree
import os
import trimesh
import open3d as o3d
from sklearn.decomposition import PCA

class InferencePipeLine:
    def __init__(self, model, pca=False):
        self.model = model
        self.pca = pca
        self.scaler = 1.8
        self.shifter = 0.8

    def __call__(self, mesh):
        if isinstance(mesh, str):
            tri_mesh_loaded_mesh = trimesh.load_mesh(mesh, process=False)
            vertex_ls = np.array(tri_mesh_loaded_mesh.vertices)
            tri_ls = np.array(tri_mesh_loaded_mesh.faces)+1

            if self.pca:
                print("PCA")
                vertex_ls = PCA(n_components=3).fit_transform(vertex_ls)
            else:
                print("No PCA")
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
            mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls)-1)
            mesh.compute_vertex_normals()
        vertices = np.array(mesh.vertices)
        n_vertices = vertices.shape[0]
        vertices[:,:3] -= np.mean(vertices[:,:3], axis=0)
        vertices[:, :3] = ((vertices[:, :3]-np.min(vertices[:,1]))/(np.max(vertices[:,1])- np.min(vertices[:,1])))*self.scaler-self.shifter
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        org_feats = np.array(np.concatenate([np.array(mesh.vertices), np.array(mesh.vertex_normals)], axis=1))

        if np.asarray(mesh.vertices).shape[0] < 24000:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        vertices = np.array(np.concatenate([np.array(mesh.vertices), np.array(mesh.vertex_normals)], axis=1))
        sampled_feats = gu.resample_pcd(vertices.copy(), 24000) #TODO slow processing speed
        with torch.no_grad():
            input_cuda_feats = torch.from_numpy(np.array([sampled_feats.astype('float32')])).cuda().permute(0,2,1)
            cls_pred = self.model(input_cuda_feats)['cls_pred']
        cls_pred = cls_pred.argmax(axis=1)
        cls_pred = gu.torch_to_numpy(cls_pred)
        cls_pred[cls_pred>=9] += 2
        cls_pred[cls_pred>0] += 10
        
        tree = KDTree(sampled_feats[:,:3], leaf_size=2)
        near_points = tree.query(org_feats[:,:3], k=1, return_distance=False)
        result_ins_labels = cls_pred.reshape(-1)[near_points.reshape(-1)].reshape(-1,1)

        return {
            "sem":result_ins_labels.reshape(-1),
            "ins":result_ins_labels.reshape(-1),
        }       
