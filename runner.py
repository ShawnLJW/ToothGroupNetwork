from generator import DentalModelGenerator
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import os
import torch


def eval_model(preds):
    pred_labels = preds.predictions.argmax(axis=1).reshape(-1)
    gt_labels = preds.label_ids.reshape(-1) + 1

    acc = accuracy_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels, average="micro")
    iou = jaccard_score(gt_labels, pred_labels, average="micro")

    return {
        "accuracy": acc,
        "f1": f1,
        "iou": iou,
    }

def collate_fn(batch):
    output = {}

    for batch_item in batch:
        for key in batch_item.keys():
            if key not in output:
                output[key] = []
            output[key].append(batch_item[key])
    
    for output_key in output.keys():
        if output_key in ["feat", "gt_seg_label", "uniform_feat", "uniform_gt_seg_label"]:
            output[output_key] = torch.stack(output[output_key])
    return output

def get_generator_set(config):
    train_ds = DentalModelGenerator(
        config["input_data_dir_path"],
        aug_obj_str=config["aug_obj_str"],
        split_with_txt_path=config["train_data_split_txt_path"],
    )

    val_ds = DentalModelGenerator(
        config["input_data_dir_path"],
        aug_obj_str=None,
        split_with_txt_path=config["val_data_split_txt_path"],
    )
    
    return train_ds, val_ds
    

def runner(config, model):
    os.environ["WANDB_PROJECT"] = "teeth_segmentation"
    train_ds, val_ds = get_generator_set(config["generator"])
    args = TrainingArguments(
        "runs",
        eval_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=1e-4,
        lr_scheduler_type="constant",
        dataloader_num_workers=4,
        num_train_epochs=30,
        load_best_model_at_end=True,
        metric_for_best_model="iou",
        report_to="wandb",
        run_name="pointnetpp-constantlr",
        logging_steps=50,
    )
    trainer = Trainer(
        model,
        args,
        data_collator=collate_fn,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=eval_model,
    )
    trainer.train()
    