from .vit import VisionTransformer

def make_backbone(config, ):
    if config.backbone.name == 'face_vit_base':
        img_size = config.input_size[0]
        patch_side = img_size // config.backbone.dynamic_patch_base_size
        num_patches = patch_side ** 2
        backbone = VisionTransformer(img_size=img_size,
                                     patch_size=patch_side,
                                     num_patches=num_patches,
                                     embed_dim=config.backbone.emb_dim,
                                     depth=24,
                                     mlp_ratio=3, num_heads=16, drop_path_rate=0.1, norm_layer="ln",
                                     mask_ratio=config.mask_ratio)
    else:
        raise NotImplementedError('Unknown backbone name')

    return backbone