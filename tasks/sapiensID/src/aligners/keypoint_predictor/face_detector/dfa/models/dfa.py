from .net import MobileNetV1, FPN, SSH, ClassHead, BboxHead, LandmarkHead
import torchvision.models._utils as _utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.models import mlp_mixer


class DifferentiableFaceAligner(nn.Module):

    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(DifferentiableFaceAligner, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
        elif cfg['name'] == 'mobilenetv4':
            from .mobilenet_v4 import MobileNetV4
            backbone = MobileNetV4(cfg['size'])
        elif cfg['name'] == 'Resnet50':
            backbone = torchvision.models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_list = cfg['return_channels']
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

        modules = [mlp_mixer.MixerBlock(16, 1050) for _ in range(3)]
        modules.append(nn.Linear(16, 1))
        self.aggregator = nn.Sequential(*modules)

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs, priorbox):

        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        decoded_bbox = priorbox.decode_batch(bbox_regressions)
        decoded_ldmk = priorbox.decode_landm_batch(ldm_regressions)
        combined = torch.cat([decoded_bbox, classifications, decoded_ldmk], dim=2)
        weight = self.aggregator(combined)
        weight = F.softmax(weight, dim=1)
        agg = torch.sum(weight * combined, dim=1)
        theta = None

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions, agg, theta)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions, agg, theta)
        return output