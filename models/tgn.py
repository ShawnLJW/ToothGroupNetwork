import torch
import numpy as np
import ops_utils as ou
import gen_utils as gu
from . import tgn_loss
from .modules.cbl_point_transformer.cbl_point_transformer_module import get_model


class GroupingNetworkModule(torch.nn.Module):
    def __init__(
        self,
        input_feat=6,
        stride=[1, 4, 4, 4, 4],
        nsample=[36, 24, 24, 24, 24],
        blocks=[2, 3, 4, 6, 3],
        block_num=5,
        planes=[32, 64, 128, 256, 512],
        crop_sample_size=3072,
    ):
        self.model_parameter = {
            "input_feat": input_feat,
            "stride": stride,
            "nsample": nsample,
            "blocks": blocks,
            "block_num": block_num,
            "planes": planes,
            "crop_sample_size": crop_sample_size,
        }

        super().__init__()
        class_num = 9
        self.first_ins_cent_model = get_model(
            **self.model_parameter,
            c=self.model_parameter["input_feat"],
            k=class_num + 1,
        )
        self.second_ins_cent_model = (
            get_model(
                **self.model_parameter,
                c=self.model_parameter["input_feat"],
                k=2,
            )
            .train()
            .cuda()
        )

    def forward(self, feat, gt_seg_label=None):
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1] => B, 1, 24000 : ground truth segmentation
        """
        B, C, N = feat.shape
        outputs = {}
        if gt_seg_label is not None:
            half_seg_label = gt_seg_label.clone()
            half_seg_label[half_seg_label >= 9] -= 8
            cbl_loss_1, sem_1, offset_1, mask_1, first_features = (
                self.first_ins_cent_model([feat, half_seg_label])
            )
            outputs.update(
                {
                    "cbl_loss_1": cbl_loss_1,
                    "sem_1": sem_1,
                    "offset_1": offset_1,
                    "mask_1": mask_1,
                    "first_features": first_features,
                }
            )
        else:
            sem_1, offset_1, mask_1, first_features = self.first_ins_cent_model([feat])
            outputs.update(
                {
                    "sem_1": sem_1,
                    "offset_1": offset_1,
                    "mask_1": mask_1,
                    "first_features": first_features,
                }
            )

        cluster_centroids = []
        if gt_seg_label is not None:
            for b_idx in range(B):
                b_gt_seg_labels = gu.torch_to_numpy(gt_seg_label[b_idx, :, :].view(-1))
                b_points_coords = gu.torch_to_numpy(feat[b_idx, :3, :]).T
                contained_tooth_num = np.unique(b_gt_seg_labels)
                temp_list = []
                for tooth_num in contained_tooth_num:
                    if tooth_num == -1:
                        continue
                    temp_list.append(
                        b_points_coords[tooth_num == b_gt_seg_labels].mean(axis=0)
                    )
                cluster_centroids.append(temp_list)
        else:
            for b_idx in range(B):
                whole_pd_sem_1 = gu.torch_to_numpy(sem_1)[b_idx, :, :].T
                whole_cls_1 = np.argmax(whole_pd_sem_1, axis=1)
                whole_offset_1 = gu.torch_to_numpy(offset_1)[b_idx, :, :].T
                b_points_coords = gu.torch_to_numpy(feat[b_idx, :3, :]).T
                b_moved_points = b_points_coords + whole_offset_1
                b_fg_moved_points = b_moved_points[whole_cls_1.reshape(-1) != 0, :]

                fg_points_labels_ls = ou.get_clustering_labels(
                    b_moved_points, whole_cls_1
                )
                temp_centroids = []
                for i in np.unique(fg_points_labels_ls):
                    temp_centroids.append(
                        np.mean(b_fg_moved_points[fg_points_labels_ls == i, :], axis=0)
                    )
                cluster_centroids.append(temp_centroids)

        org_xyz_cpu = gu.torch_to_numpy(feat[:, :3, :].permute(0, 2, 1))
        nn_crop_indexes = ou.get_nearest_neighbor_idx(
            org_xyz_cpu,
            cluster_centroids,
            self.model_parameter["crop_sample_size"],
        )
        cropped_feature_ls = ou.get_indexed_features(feat, nn_crop_indexes)
        if gt_seg_label is not None:
            cluster_gt_seg_label = ou.get_indexed_features(
                gt_seg_label, nn_crop_indexes
            )

        cropped_feature_ls = ou.centering_object(cropped_feature_ls)
        if gt_seg_label is not None:
            cluster_gt_seg_label[cluster_gt_seg_label >= 0] = 0
            outputs["cluster_gt_seg_label"] = cluster_gt_seg_label
            cbl_loss_2, sem_2, offset_2, mask_2, _ = self.second_ins_cent_model(
                [cropped_feature_ls, cluster_gt_seg_label]
            )

            outputs.update(
                {
                    "cbl_loss_2": cbl_loss_2,
                    "sem_2": sem_2,
                    "offset_2": offset_2,
                    "mask_2": mask_2,
                }
            )
        else:
            sem_2, offset_2, mask_2, _ = self.second_ins_cent_model(
                [cropped_feature_ls]
            )
            outputs.update(
                {
                    "sem_2": sem_2,
                    "offset_2": offset_2,
                    "mask_2": mask_2,
                }
            )

        outputs["cropped_feature_ls"] = cropped_feature_ls
        outputs["nn_crop_indexes"] = nn_crop_indexes

        if gt_seg_label is not None:
            half_seg_label = gt_seg_label.clone()
            half_seg_label[half_seg_label >= 9] -= 8

            cluster_gt_seg_label[cluster_gt_seg_label >= 0] = 0
            tooth_class_loss_1 = tgn_loss.tooth_class_loss(sem_1, half_seg_label, 9)
            tooth_class_loss_2 = tgn_loss.tooth_class_loss(
                sem_2, cluster_gt_seg_label, 2
            )

            offset_1_loss, offset_1_dir_loss = tgn_loss.batch_center_offset_loss(
                offset_1, feat[:, :3, :], gt_seg_label
            )

            chamf_1_loss = tgn_loss.batch_chamfer_distance_loss(
                offset_1, feat[:, :3, :], gt_seg_label
            )

            outputs["loss"] = (
                cbl_loss_1.sum()
                + cbl_loss_2.sum()
                + tooth_class_loss_1
                + tooth_class_loss_2
                + offset_1_loss * 0.03
                + offset_1_dir_loss * 0.03
                + chamf_1_loss * 0.15
            )
        return outputs
