import gen_utils as gu
import numpy as np
import pandas as pd
from tqdm import trange
from inference_pipelines.inference_pipeline_maker import make_inference_pipeline

# change the following paths to the correct paths
with open("base_name_test_fold.txt", "r") as f:
    ids = [x for x in f.read().split("\n") if x != ""]
scan_list = []
gt_list = []
for id in ids:
    for jaw in ["upper", "lower"]:
        scan_list.append(f"data_obj_parent_directory/{id}/{id}_{jaw}.obj")
        gt_list.append(f"data_json_parent_directory/{id}/{id}_{jaw}.json")
    
def cal_metric(
    gt_labels, pred_sem_labels, pred_ins_labels, is_half=None, vertices=None
):
    ins_label_names = np.unique(pred_ins_labels)
    ins_label_names = ins_label_names[ins_label_names != 0]
    IOU = 0
    F1 = 0
    ACC = 0
    SEM_ACC = 0
    IOU_arr = []
    for ins_label_name in ins_label_names:
        # instance iou
        ins_label_name = int(ins_label_name)
        ins_mask = pred_ins_labels == ins_label_name
        gt_label_uniqs, gt_label_counts = np.unique(
            gt_labels[ins_mask], return_counts=True
        )
        gt_label_name = gt_label_uniqs[np.argmax(gt_label_counts)]
        gt_mask = gt_labels == gt_label_name

        TP = np.count_nonzero(gt_mask * ins_mask)
        FN = np.count_nonzero(gt_mask * np.invert(ins_mask))
        FP = np.count_nonzero(np.invert(gt_mask) * ins_mask)
        TN = np.count_nonzero(np.invert(gt_mask) * np.invert(ins_mask))

        ACC += (TP + TN) / (FP + TP + FN + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 += 2 * (precision * recall) / (precision + recall)
        IOU += TP / (FP + TP + FN)
        IOU_arr.append(TP / (FP + TP + FN))
        # segmentation accuracy
        pred_sem_label_uniqs, pred_sem_label_counts = np.unique(
            pred_sem_labels[ins_mask], return_counts=True
        )
        sem_label_name = pred_sem_label_uniqs[np.argmax(pred_sem_label_counts)]
        if is_half:
            if sem_label_name == gt_label_name or sem_label_name + 8 == gt_label_name:
                SEM_ACC += 1
        else:
            if sem_label_name == gt_label_name:
                SEM_ACC += 1
    return (
        IOU / len(ins_label_names),
        F1 / len(ins_label_names),
        ACC / len(ins_label_names),
        SEM_ACC / len(ins_label_names),
        IOU_arr,
    )
    
def evaluate_model(model_name, pca=False):
    print(f"evaluating {model_name}{'' if not pca else '_pca'}")
    IoU_total, F1_total, Acc_total, SEM_ACC_total, n = 0, 0, 0, 0, 0
    pipeline = make_inference_pipeline(model_name, [f"ckpts/{model_name}.h5"], pca=pca)

    for i in trange(len(scan_list)):
        preds = pipeline(scan_list[i])
        gt = gu.load_json(gt_list[i])
        gt = np.array(gt["labels"])

        IoU, F1, Acc, SEM_ACC, _ = cal_metric(
            gt, preds["sem"], preds["ins"], is_half=True
        )
        IoU_total += IoU
        F1_total += F1
        Acc_total += Acc
        SEM_ACC_total += SEM_ACC
        n += 1

    return {
        "model_name": f"{model_name}{'' if not pca else '_pca'}",
        "IoU": IoU_total / n,
        "F1": F1_total / n,
        "Acc": Acc_total / n,
        "SEM_ACC": SEM_ACC_total / n,
    }
    
models = [
    "dgcnn",
    "pointnet",
    "pointnetpp",
    "pointtransformer",
]
results = pd.DataFrame([evaluate_model(model) for model in models] + [evaluate_model(model, pca=True) for model in models])
results.to_csv("results.csv", index=False)