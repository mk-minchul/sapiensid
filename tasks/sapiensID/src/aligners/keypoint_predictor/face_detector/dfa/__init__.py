from .models.retinaface import RetinaFace
from .utils.model_utils import load_model
from .config import cfg_mnet, cfg_re50
from .layers.functions.prior_box import PriorBox
from .preprocessor import Preprocessor
from .models.dfa import DifferentiableFaceAligner
import torch
import torch.nn as nn
import os

class FaceKPPredictor(nn.Module):
    def __init__(self, net, prior_box, preprocessor):
        super(FaceKPPredictor, self).__init__()
        self.net = net
        self.prior_box = prior_box
        self.preprocessor = preprocessor

    def forward(self, unnorm_images_rgb):
        face_input = (unnorm_images_rgb - 0.5) / 0.5
        face_input = self.preprocessor(face_input)
        face_input = face_input.flip(1)
        result = self.net(face_input, self.prior_box)
        anchor_bbox_pred, anchor_cls_pred, anchor_ldmk_pred, merged, _ = result
        bbox, cls, ldmk = torch.split(merged, [4, 2, 10], dim=1)
        bbox = bbox.clamp(min=0, max=1)

        face_score = torch.nn.Softmax(dim=-1)(cls)[:,1:]
        ldmk = ldmk.view(-1, ldmk.size(1)//2, 2)
        return face_score, bbox, ldmk


def get_landmark_predictor(network='mobile0.25', use_aggregator=True, input_size=160):

    cfg = None
    if network == "mobile0.25":
        cfg = cfg_mnet
        net = RetinaFace(cfg=cfg, phase = 'test', use_aggregator=use_aggregator)
    elif network == "resnet50":
        cfg = cfg_re50
        net = RetinaFace(cfg=cfg, phase = 'test', use_aggregator=use_aggregator)
    elif network == 'mobilenetv4':
        cfg = {
            'name': 'mobilenetv4',
            'size': 'MobileNetV4ConvMedium',
            'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
            'return_channels': [80, 160, 256],
            # 2: 80 3: 160 4: 256, 5: 1280
            'out_channel': 64
        }
        net = DifferentiableFaceAligner(cfg=cfg, phase='test')

    priorbox = PriorBox(image_size=(input_size, input_size),
                        min_sizes=[[64, 80], [96, 112], [128, 144]],
                        steps=[8, 16, 32],
                        clip=False,
                        variances=[0.1, 0.2],)

    return net, priorbox

def get_preprocessor(output_size=160, padding=0.0, padding_val='zero'):
    return Preprocessor(output_size=output_size, padding=padding, padding_val=padding_val)

def get_face_kp_predictor(pretrain_path='../../pretrained_models/aligners/dfa_mobilenetv4_medium/mobilenetv4_Final.pth'):
    net, prior_box = get_landmark_predictor(
        # network='mobile0.25',
        network='mobilenetv4',
        use_aggregator=True,
        input_size=160)

    preprocessor = get_preprocessor(output_size=160,
                                    padding=0,
                                    padding_val='zero')
    for param in net.parameters():
        param.requires_grad = False
    state_dict = torch.load(pretrain_path)
    net.load_state_dict(state_dict)
    aligner = FaceKPPredictor(net, prior_box, preprocessor)
    return aligner
