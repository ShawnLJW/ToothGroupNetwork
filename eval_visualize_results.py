import argparse
import gen_utils as gu
import numpy as np
from inference_pipelines.inference_pipeline_maker import make_inference_pipeline
from metrics import calculate_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--labels_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="tgnet")
    parser.add_argument(
        "--ckpt", default="ckpts/tgnet_fps.h5", help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--ckpt_bdl",
        default="ckpts/tgnet_bdl.h5",
        help="Path to the checkpoint file for tgnet BDL module",
    )
    args = parser.parse_args()

    pipeline = make_inference_pipeline(
        model_name=args.model_name,
        ckpt_path=args.ckpt,
        bdl_ckpt_path=args.ckpt_bdl,
    )

    feats, mesh = gu.load_mesh(args.input_path)
    vertices = feats[:, :3]
    outputs = pipeline(mesh)
    mesh = gu.get_colored_mesh(mesh, outputs["sem"])

    if args.labels_path:
        labels = gu.load_labels(args.labels_path)
        metrics = calculate_metrics(
            vertices,
            labels,
            labels,
            outputs["ins"],
            outputs["sem"],
        )
        print(metrics)

    gu.print_3d(mesh)
