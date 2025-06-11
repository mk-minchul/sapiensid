import torch
from torch import nn
import torch.nn.functional as F
from general_utils.img_utils import visualize, pil_to_tensor
import os
import numpy as np

from .dynamic_patching import  make_dynamic_patch_v2, \
    visualize_patches, make_dynamic_pos_embed, visualize_masked_patch, \
    visualize_pos_embed_v2, visualize_patch_overlay, visualize_masked_patch_with_ids_keep

from .dynamic_patching_utils import unfold_mask, snap_bbox_to_patch, \
        warp_bbox_space_from_orig_to_body, warp_bbox_space_from_body_to_original, \
        masked_sampling, get_square_pos_embed, add_level_emb

INVALID_KPRPE_EMB_VALUE = -2

class DynamicPather(nn.Module):

    def __init__(self, use_fg_mask, use_mask_token, dynamic_patch_base_size, emb_dim, level_emb=False):
        
        super(DynamicPather, self).__init__()
        
        in_feature = dynamic_patch_base_size ** 2 * 3
        self.patch_embed = nn.Linear(in_features=in_feature, out_features=emb_dim, bias=True)
        self.patch_embed.bias.data = torch.zeros_like(self.patch_embed.bias.data)

        self.use_fg_mask = use_fg_mask
        self.use_mask_token = use_mask_token
        self.dynamic_patch_base_size = dynamic_patch_base_size
        self.level_emb = level_emb

        if self.level_emb:
            self.level_embed = nn.Parameter(torch.zeros(3, emb_dim))
            nn.init.trunc_normal_(self.level_embed, std=0.02)


    def forward(self, images, ldmks, body_bboxes, face_bboxes, backbone, 
                foreground_masks=None, subset_method=None):
        B, C, H, W = images.shape
        base_patch_size = self.dynamic_patch_base_size

        # create foreground mask as tokens
        if self.use_fg_mask and foreground_masks is not None:
            foreground_masks = unfold_mask(foreground_masks, (base_patch_size, base_patch_size), (H, W))
        else:
            assert not self.training, "Foreground masks must be provided during training"
            n_patches = (H // base_patch_size) * (W // base_patch_size)
            foreground_masks = torch.ones((B, n_patches), device=images.device, dtype=torch.bool)

        # make it snap to the nearest patch
        n_side = H // base_patch_size
        body_bboxes = snap_bbox_to_patch(body_bboxes, n_side)
        face_bboxes_in_body = warp_bbox_space_from_orig_to_body(face_bboxes, body_bboxes)
        face_bboxes_in_body = snap_bbox_to_patch(face_bboxes_in_body, n_side)
        face_bboxes = warp_bbox_space_from_body_to_original(face_bboxes_in_body, body_bboxes)

        # make dynamic patch
        dynamic_patches = make_dynamic_patch_v2(images, face_bboxes, body_bboxes, foreground_masks, base_patch_size=base_patch_size)
        
        # make pos embed for dynamic patches
        output_size = (H//base_patch_size, W//base_patch_size)
        pos_cls_embed, square_pos_embed = get_square_pos_embed(backbone)
        dynamic_pos_embed = make_dynamic_pos_embed(square_pos_embed, face_bboxes, body_bboxes, output_size)

        if self.level_emb:
            dynamic_pos_embed = add_level_emb(dynamic_pos_embed, self.level_embed)

        # test if dynamic patches and pos embed are correctly sampled
        debug_vis = hasattr(backbone, 'debug_vis') and backbone.debug_vis
        # masked sampling
        x_masked, mask, ids_restore, ids_keep, pos_embed_masked, \
            imp_mask_masked, full_importance_mask = masked_sampling(dynamic_patches,
                                                                    dynamic_pos_embed,
                                                                    mask_ratio=backbone.mask_ratio,
                                                                    is_train=self.training,
                                                                    subset_method=subset_method,
                                                                    ldmks=ldmks)
        

        if debug_vis:
            os.makedirs('dp_vis', exist_ok=True)
            # dynamic_patches = [dynamic_patches[-5]]
            # dynamic_pos_embed = {key:val[-5:-4] for key, val in dynamic_pos_embed.items()}
            # face_bboxes_in_body = face_bboxes_in_body[-5:-4]
            # body_bboxes = body_bboxes[-5:-4]
            # images = images[-5:-4]
            # ldmks = ldmks[-5:-4]
            # x_masked = x_masked[-5:-4]
            # ids_restore = ids_restore[-5:-4]
            num_vis_patch_existing = len([f for f in os.listdir('dp_vis') if 'vis_patch_' in f])
            num_vis_patch_emb_existing = len([f for f in os.listdir('dp_vis') if 'vis_pos_' in f])
            vis_patch = visualize_patches(dynamic_patches,
                                          save_name=f'dp_vis/vis_patch_{num_vis_patch_existing}.png')
            vis_patch = visualize_patches(dynamic_patches,
                                          save_name=f'dp_vis/vis_patch_{num_vis_patch_existing}.png', fade_non_important=False)
            vis_patch_emb = visualize_pos_embed_v2(dynamic_patches, dynamic_pos_embed,
                                                   save_name=f'dp_vis/vis_pos_{num_vis_patch_emb_existing}.png')
            vis_masked = visualize_masked_patch(base_patch_size, x_masked, ids_restore, images, 
                                                save_name=f'dp_vis/vis_masked_{num_vis_patch_emb_existing}.png')
            _pos_embed_masked = pos_embed_masked[:, :, :, None, None].repeat(1, 1, 1, base_patch_size, base_patch_size).flatten(2)
            _pos_embed_masked = ((_pos_embed_masked / 255) - 0.5) / 0.5
            vis_pos_embed_masked = visualize_masked_patch(base_patch_size, _pos_embed_masked, ids_restore, images, 
                                                save_name=f'dp_vis/vis_masked_{num_vis_patch_emb_existing}_pos_embed.png')
            vis_orig = pil_to_tensor(visualize(images.cpu(), axis=0)).unsqueeze(0)
            vis_ldmk = pil_to_tensor(visualize(images.cpu(), ldmks.cpu(), axis=0)).unsqueeze(0)
            vis = torch.cat([vis_orig, vis_ldmk, vis_masked, vis_patch, vis_patch_emb], dim=3).cpu()

            visualize(vis).save(f'dp_vis/vis_{num_vis_patch_emb_existing}.png')

            # sub pathces are overlayed to the bigger image
            all_overlay = visualize_patch_overlay(vis_patch, face_bboxes_in_body, body_bboxes)
            visualize(all_overlay.cpu()).save(f'dp_vis/vis_warp_{num_vis_patch_emb_existing}.png')
            visualize(all_overlay[:, :, :, :384]).save(f'dp_vis/vis_warp_one{num_vis_patch_emb_existing}.png')
            all_overlay = visualize_patch_overlay(vis_patch_emb, face_bboxes_in_body, body_bboxes)
            visualize(all_overlay.cpu()).save(f'dp_vis/vis_warp_emb_{num_vis_patch_emb_existing}.png')
            all_overlay_masked = visualize_patch_overlay(vis_masked, face_bboxes_in_body, body_bboxes)
            visualize(all_overlay_masked.cpu()).save(f'dp_vis/vis_warp_masked_{num_vis_patch_emb_existing}.png')

            visualize_masked_patch_with_ids_keep(x_masked, ids_restore, ids_keep, base_patch_size, 
                                                 H, W, save_name=f'dp_vis/vis_masked_with_ids_{num_vis_patch_emb_existing}.png')

            visualize(vis_orig).save(f'dp_vis/vis_orig_{num_vis_patch_emb_existing}.png')
            visualize(vis_ldmk).save(f'dp_vis/vis_ldmk_{num_vis_patch_emb_existing}.png')
            visualize(vis_patch[:, :, :, :384]).save(f'dp_vis/vis_patch1_{num_vis_patch_emb_existing}.png')
            visualize(vis_patch[:, :, :, 384:384*2]).save(f'dp_vis/vis_patch2_{num_vis_patch_emb_existing}.png')
            visualize(vis_patch[:, :, :, 384*2:]).save(f'dp_vis/vis_patch3_{num_vis_patch_emb_existing}.png')
            visualize(vis_patch_emb[:, :, :, :384]).save(f'dp_vis/vis_patch_emb1_{num_vis_patch_emb_existing}.png')
            visualize(vis_patch_emb[:, :, :, 384:384*2]).save(f'dp_vis/vis_patch_emb2_{num_vis_patch_emb_existing}.png')
            visualize(vis_patch_emb[:, :, :, 384*2:]).save(f'dp_vis/vis_patch_emb3_{num_vis_patch_emb_existing}.png')
            visualize(all_overlay_masked[:, :, :, :384]).save(f'dp_vis/vis_warp_masked1_{num_vis_patch_emb_existing}.png')
            vis_patch = visualize_patches(dynamic_patches,
                                          save_name=f'dp_vis/vis_patch_{num_vis_patch_existing}.png', fade_non_important=False)
            visualize(vis_orig).save(f'dp_vis/vis_orig_{num_vis_patch_emb_existing}.png')
            visualize(vis_ldmk).save(f'dp_vis/vis_ldmk_{num_vis_patch_emb_existing}.png')
            visualize(vis_patch[:, :, :, :384]).save(f'dp_vis/vis_patch1_{num_vis_patch_emb_existing}.png')
            visualize(vis_patch[:, :, :, 384:384*2]).save(f'dp_vis/vis_patch2_{num_vis_patch_emb_existing}.png')
            visualize(vis_patch[:, :, :, 384*2:]).save(f'dp_vis/vis_patch3_{num_vis_patch_emb_existing}.png')
            visualize(vis_patch_emb[:, :, :, :384]).save(f'dp_vis/vis_patch_emb1_{num_vis_patch_emb_existing}.png')
            visualize(vis_patch_emb[:, :, :, 384:384*2]).save(f'dp_vis/vis_patch_emb2_{num_vis_patch_emb_existing}.png')
            visualize(vis_patch_emb[:, :, :, 384*2:]).save(f'dp_vis/vis_patch_emb3_{num_vis_patch_emb_existing}.png')

            print('vis saved')


        x = self.patch_embed(x_masked)

        return {
            'x': x,
            'mask': mask,
            'pos_embed_masked': pos_embed_masked,
            'pos_cls_embed': pos_cls_embed,
            'imp_mask_masked': imp_mask_masked,
            'full_importance_mask': full_importance_mask,
            'ids_keep': ids_keep,
            'ids_restore': ids_restore,
            'square_pos_embed': square_pos_embed
        }




