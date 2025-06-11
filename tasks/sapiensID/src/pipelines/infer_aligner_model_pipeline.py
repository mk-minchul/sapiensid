from .base import BasePipeline
from src.models.base import BaseModel
from transformers.modeling_outputs import ModelOutput
import torch


class InferAlignerModelPipeline(BasePipeline):

    def __init__(self,
                 model:BaseModel,
                 aligner:BaseModel=None,
                 ):
        super(InferAlignerModelPipeline, self).__init__()

        self.model = model
        self.aligner = aligner
        self.eval()

        self._theta_cache = {}

    @property
    def module_names_list(self):
        return ['model', ]

    def integrity_check(self, dataset_color_space):
        # color space check
        assert dataset_color_space == self.model.config.color_space
        self.color_space = dataset_color_space
        self.make_test_transform()

    def make_test_transform(self):
        return self.model.make_test_transform()


    def __call__(self, batch):
        input_tensor = batch
        inner_bs = 16
        input_tensor_splits = input_tensor.split(inner_bs)
        feats = []
        for patches in input_tensor_splits:
            keypoints, foreground_masks = self.aligner(patches)
            feat = self.get_feature(self.model(patches, foreground_masks=foreground_masks, ldmks=keypoints))
            feats.append(feat)
        feats = torch.cat(feats, dim=0)
        return feats

    def get_feature(self, output):
        if isinstance(output, ModelOutput):
            feature = output['pooler_output']
        else:
            feature = output
        return feature


    def train(self):
        raise NotImplementedError('InferAlignerModelPipeline does not support train mode')


    def eval(self):
        self.model.eval()
        if self.aligner:
            self.aligner.eval()


