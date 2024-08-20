import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from models.tgn_loss import tooth_class_loss
from pointnet_utils import PointNetEncoder


class PointNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k = 17
        scale = 2
        self.feat = PointNetEncoder(
            global_feat=False, feature_transform=True, channel=6, scale=scale
        )
        self.conv1 = torch.nn.Conv1d(1088 * scale, 512 * scale, 1)
        self.conv2 = torch.nn.Conv1d(512 * scale, 256 * scale, 1)
        self.conv3 = torch.nn.Conv1d(256 * scale, 128 * scale, 1)
        self.conv4 = torch.nn.Conv1d(128 * scale, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512 * scale)
        self.bn2 = nn.BatchNorm1d(256 * scale)
        self.bn3 = nn.BatchNorm1d(128 * scale)

    def forward(self, feat, gt_seg_label=None):
        """
        Args:
            inputs (B, 6, 24000): point features

        Returns:
            cls_pred (B, 17, 24000): predicted class
        """
        batchsize = feat.size()[0]
        n_pts = feat.size()[2]
        x, trans, trans_feat = self.feat(feat)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        cls_pred = F.log_softmax(x.view(-1, self.k), dim=-1)
        cls_pred = cls_pred.view(batchsize, n_pts, self.k).permute(0, 2, 1)
        outputs = {"cls_pred": cls_pred}
        if gt_seg_label is not None:
            outputs["loss"] = tooth_class_loss(cls_pred, gt_seg_label, 17)
        return outputs
