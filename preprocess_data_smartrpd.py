import os
import numpy as np
import gen_utils as gu


def read_txt_labels(path):
    labels = [int(x) for x in f.read().split("\n") if x != ""]
    labels = np.array(labels)
    labels[labels >= 9] = 7 - labels[labels >= 9]
    labels[labels != 0] = 19 - labels[labels != 0]
    
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
        labels = read_txt_labels(label_path)
        labeled_vertices = np.concatenate([feats, labels.reshape(-1,1)], axis=1)
        labeled_vertices = gu.resample_pcd(labeled_vertices, 24000)
        np.save(f"{output_dir}/{id}_sampled_points", labeled_vertices)