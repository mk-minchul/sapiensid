from ..base import BaseModel
from torchvision import transforms
import torchvision.transforms.functional as F
from .heads import make_head
from .mrm import MaskedRecognitionModel
from .backbones.vit_face import make_backbone as make_backbone_face_vit
from .rpe import make_rpe
from .preproc_backbone import preproc_patch_emb, preproc_cls_token, preproc_pos_emb
from .patching import make_patch_embed
from .kp_preproc import make_kp_preprocessor


class SapiensModel(BaseModel):

    def __init__(self, config):
        super(SapiensModel, self).__init__(config)

        assert config.input_size[0] == config.input_size[1], 'Only square input size is supported'

        if config.backbone.name == 'face_vit_base':
            backbone = make_backbone_face_vit(config)
            depth = 24
            num_heads = 16
        else:
            raise NotImplementedError

        backbone = preproc_patch_emb(backbone, 
                                     config.backbone.emb_dim, 
                                     config.backbone.dynamic_patch_base_size)
        backbone = preproc_cls_token(backbone, config.backbone.emb_dim)
        backbone = preproc_pos_emb(backbone,
                                   config.backbone.pos_embed_type,
                                   config.backbone.dynamic_patch_base_size,
                                   config.input_size)

        # create patch emb
        patch_embed = make_patch_embed(config.patch_embed_config)

        # create kp preprocessor
        kp_prprocessor = make_kp_preprocessor(config)

        # create head
        head = make_head(config.head)

        # create KPRPE
        rpe_module = make_rpe(config.rpe_config, depth, num_heads, config.backbone.emb_dim)

        self.net = MaskedRecognitionModel(backbone=backbone,
                                            kp_prprocessor=kp_prprocessor,
                                            patch_embed=patch_embed,
                                            head=head,
                                            rpe_module=rpe_module,
                                            config=config)

        self.config = config
        self.input_size = config.input_size # (height, width)
        self.rgb_mean = config.rgb_mean
        self.rgb_std = config.rgb_std
        self.square_pad = config.square_pad


    @classmethod
    def from_config(cls, config):
        model = cls(config)
        model.eval()
        return model

    def forward(self, x, foreground_masks=None, ldmks=None, return_intermediate=False, **kwargs):
        if self.input_color_flip:
            x = x.flip(1)
        return self.net(x, foreground_masks, ldmks, return_intermediate=return_intermediate, **kwargs)

    def make_train_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.input_size, antialias=None),
            transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std),
        ])
        return transform

    def make_test_transform(self):
        if self.square_pad:
            transform = transforms.Compose([
                transforms.ToTensor(),
                SquarePad(fill=1),
                transforms.Resize(self.input_size, antialias=None),
                transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.input_size, antialias=None),
                transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std),
            ])
        return transform

def load_model(model_config):
    model = SapiensModel.from_config(model_config)
    return model



class SquarePad:
    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        c, h, w = img.shape
        max_wh = max(w, h)
        hp = max_wh - h
        wp = max_wh - w
        padding = (wp // 2, hp // 2, wp - (wp // 2), hp - (hp // 2))
        return F.pad(img, padding, self.fill, self.padding_mode)


