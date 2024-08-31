import argparse
import os
import torch
from generator import DentalModelGenerator
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, jaccard_score


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
        if output_key in [
            "feat",
            "gt_seg_label",
            "uniform_feat",
            "uniform_gt_seg_label",
        ]:
            output[output_key] = torch.stack(output[output_key])
    return output


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "teeth_segmentation"
    os.environ["WANDB_LOG_MODEL"] = "end"

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--model_name", type=str)

    # augmentation parameters
    parser.add_argument("--scaling_range", nargs="*", type=float, default=[0.85, 1.15])
    parser.add_argument("--rotation_range", nargs="*", type=float, default=[-30, 30])
    parser.add_argument("--rotation_axis", type=str, default="fixed")
    parser.add_argument(
        "--translation_range", nargs="*", type=float, default=[-0.2, 0.2]
    )

    # training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_schedule", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)

    # paths to files and directories
    parser.add_argument("--input_data_dir", type=str, default="data_preprocessed_path")
    parser.add_argument("--train_split", type=str, default="base_name_train_fold.txt")
    parser.add_argument("--val_split", type=str, default="base_name_val_fold.txt")

    args = parser.parse_args()

    if args.model_name == "dgcnn":
        from models.dgcnn import DGCnn

        model = DGCnn()
    elif args.model_name == "pointnet":
        from models.pointnet import PointNet

        model = PointNet()
    elif args.model_name == "pointnetpp":
        from models.pointnet_pp import PointNetPp

        model = PointNetPp()
    elif args.model_name == "pointmlp":
        from models.pointmlp import PointMLP

        model = PointMLP(num_channels=6, num_classes=17)
    elif args.model_name == "pointtransformer":
        from models.point_transformer import PointTransformer

        model = PointTransformer()
    else:
        raise ValueError(f"Model {args.model_name} not found")

    train_ds = DentalModelGenerator(
        data_dir=args.input_data_dir,
        scaling_range=args.scaling_range,
        rotation_range=args.rotation_range,
        rotation_axis=args.rotation_axis,
        translation_range=args.translation_range,
    )

    val_ds = DentalModelGenerator(
        data_dir=args.input_data_dir, split_with_txt_path=args.val_split
    )

    save_path = f"runs/{args.run_name}"
    training_args = TrainingArguments(
        output_dir=save_path,
        eval_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_schedule,
        dataloader_num_workers=4,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="iou",
        report_to="wandb",
        run_name=args.run_name,
        logging_steps=50,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=eval_model,
    )

    trainer.train()
    trainer.save_model(save_path)
