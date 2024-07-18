import gen_utils as gu
import numpy as np
import pandas as pd
from tqdm import trange
from inference_pipelines.inference_pipeline_maker import make_inference_pipeline
from metrics import calculate_metrics

# change the following paths to the correct paths
with open("base_name_test_fold.txt", "r") as f:
    ids = [x for x in f.read().split("\n") if x != ""]
scan_list = []
gt_list = []
for id in ids:
    for jaw in ["upper", "lower"]:
        scan_list.append(f"data_obj_parent_directory/{id}/{id}_{jaw}.obj")
        gt_list.append(f"data_json_parent_directory/{id}/{id}_{jaw}.json")
    
def evaluate_model(model_name, pca=False):
    print(f"evaluating {model_name}{'' if not pca else '_pca'}")
    pipeline = make_inference_pipeline(model_name, [f"ckpts/{model_name}.h5"])
    name = f"{model_name}{'' if not pca else '_pca'}"
    model_results = []

    for i in trange(len(scan_list)):
        feats, mesh = gu.load_mesh(scan_list[i])
        vertices = feats[:, :3]
        outputs = pipeline(mesh, pca)
        
        gt = gu.load_json(gt_list[i])
        metrics = calculate_metrics(
            vertices,
            np.array(gt["instances"]),
            np.array(gt["labels"]),
            outputs["ins"],
            outputs["sem"],
        )
        metrics["file_path"] = scan_list[i]
        model_results.append(metrics)
    
    model_results = pd.DataFrame(model_results)
    model_results.to_csv(f"results_{name}.csv", index=False)
    
    summary_results = {"model_name": name}
    summary_results.update(model_results.mean(numeric_only=True).to_dict())
    return summary_results
    
models = [
    "dgcnn",
    "pointnet",
    "pointnetpp",
    "pointtransformer",
]
results = pd.DataFrame([evaluate_model(model, pca) for model in models for pca in [False, True]])
results.to_csv("results.csv", index=False)