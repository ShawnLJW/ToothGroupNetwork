import torch
from .modules.cbl_point_transformer.cbl_point_transformer_module import get_model
from .tgn_loss import tooth_class_loss

class PointTransformer(torch.nn.Module):
    def __init__(self):

        super().__init__()
        model_parameter = {
            "input_feat": 6,
            "stride": [1, 4, 4, 4, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 4, 6, 3],
            "block_num": 5,
            "planes": [32, 64, 128, 256, 512],
            "contain_weight": False,
            "crop_sample_size": 3072,
        }
        self.first_ins_cent_model = get_model(config=model_parameter, c=6, k=17, **model_parameter)

    def forward(self, feat, gt_seg_label=None):
        sem_1, offset_1, mask_1, first_features = self.first_ins_cent_model([feat])
        outputs = {"cls_pred": sem_1}
        if gt_seg_label is not None:
            outputs["loss"] = tooth_class_loss(sem_1, gt_seg_label, 17)
        return outputs