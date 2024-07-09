import torch
from .modules.cbl_point_transformer.cbl_point_transformer_module import get_model
from .tgn_loss import tooth_class_loss

class PointTransformer(torch.nn.Module):
    def __init__(self, config):
        self.config = config

        super().__init__()
        self.first_ins_cent_model = get_model(**config["model_parameter"], c=config["model_parameter"]["input_feat"], k=16 + 1)

    def forward(self, feat, gt_seg_label=None):
        outputs = {}
        sem_1, offset_1, mask_1, first_features = self.first_ins_cent_model([feat])
        outputs.update({
            "sem_1": sem_1,
            "offset_1":offset_1,
            "mask_1":mask_1,
            "first_features": first_features,
            "cls_pred": sem_1
        })
        if gt_seg_label is not None:
            outputs["loss"] = tooth_class_loss(sem_1, gt_seg_label, 17)
        return outputs