from .dynamic_patcher import DynamicPather
from .patch_embed_conv import PatchEmbed

def make_patch_embed(patch_embed_config):
    if patch_embed_config.name == 'conv':
        patch_embed = PatchEmbed(img_size=patch_embed_config.input_size, 
                                 patch_size=patch_embed_config.patch_size, 
                                 in_channels=patch_embed_config.in_channels, 
                                 embed_dim=patch_embed_config.embed_dim)
    elif patch_embed_config.name == 'dynamic':
        if hasattr(patch_embed_config, 'level_emb'):
            level_emb = patch_embed_config.level_emb
        else:
            level_emb = False
        patch_embed = DynamicPather(use_fg_mask=patch_embed_config.use_fg_mask,
                                    use_mask_token=patch_embed_config.use_mask_token,
                                    dynamic_patch_base_size=patch_embed_config.dynamic_patch_base_size,
                                    emb_dim=patch_embed_config.emb_dim,
                                    level_emb=level_emb)
    return patch_embed

