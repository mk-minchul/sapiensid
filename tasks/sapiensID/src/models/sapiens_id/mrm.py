import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

INVALID_KPRPE_EMB_VALUE = -2

class MaskedRecognitionModel(nn.Module):

    def __init__(self, backbone, kp_prprocessor, patch_embed, head, rpe_module, config):
        super(MaskedRecognitionModel, self).__init__()
        self.backbone = backbone
        self.kp_prprocessor = kp_prprocessor
        self.patch_embed = patch_embed
        self.rpe_module = rpe_module
        self.head = head
        self.config = config

        if self.config.backbone.use_mask_token:
            self.mask_token = nn.Parameter(torch.randn(1, config.backbone.emb_dim))
            self.mask_token.requires_grad = True
            torch.nn.init.normal_(self.mask_token, std=.02)

        self.use_rpe = self.config.rpe_config is not None
        print('RPE module: use True', rpe_module)
        debug_vis = hasattr(self.backbone, 'debug_vis') and self.backbone.debug_vis
        if debug_vis:
            self.rpe_module = None

    def mrm_forward(self, images, foreground_masks=None, ldmks=None, subset_method=None):

        B = images.shape[0]

        # preproc ldmks
        ldmks, body_bboxes, face_bboxes = self.kp_prprocessor(ldmks, self.training)

        # patch embed
        patch_embed_dict = self.patch_embed(images, ldmks, body_bboxes, face_bboxes, self.backbone, foreground_masks, subset_method)
        x = patch_embed_dict['x']
        mask = patch_embed_dict['mask']
        pos_embed_masked = patch_embed_dict['pos_embed_masked']
        pos_cls_embed = patch_embed_dict['pos_cls_embed']
        imp_mask_masked = patch_embed_dict['imp_mask_masked']
        full_importance_mask = patch_embed_dict['full_importance_mask']
        ids_keep = patch_embed_dict['ids_keep']
        ids_restore = patch_embed_dict['ids_restore']
        square_pos_embed = patch_embed_dict['square_pos_embed']

        B = ldmks.shape[0]
        ldmks_grid =(ldmks * 2 - 1).unsqueeze(2).float()
        ldmk_embs = F.grid_sample(square_pos_embed.expand(B, -1, -1, -1), ldmks_grid, align_corners=True).squeeze(3)
        
        if self.use_rpe:
            rel_kp_embs = make_relative_keypoints_embed(ldmks, square_pos_embed, pos_embed_masked)
        else:
            rel_kp_embs = None

        # add pos_emb
        debug_vis = hasattr(self.backbone, 'debug_vis') and self.backbone.debug_vis
        if debug_vis:
            pos_cls_embed = 0
            pos_embed_masked = 0
        x = x + pos_embed_masked

        # add cls token and optionally mask token
        x, imp_mask_masked, full_importance_mask, rel_kp_embs = self.apply_cls_token(x, pos_cls_embed, imp_mask_masked, full_importance_mask, rel_kp_embs)

        # apply mask token
        x, effective_count, rel_kp_embs = self.apply_mask_token(x, imp_mask_masked, full_importance_mask, rel_kp_embs)

        # infer kprpe
        if self.use_rpe and not debug_vis:
            if self.config.backbone.use_mask_token:
                aux_tokens = 'cls_first_mask_last'
            else:
                aux_tokens = 'cls_first'
            patched_img_size = square_pos_embed.shape[2:]
            offset_biases = self.rpe_module(rel_kp_embs, ids_keep, aux_tokens=aux_tokens, patched_img_size=patched_img_size)
        else:
            offset_biases = None

        # forward pass
        x = self.backbone.forward_blocks(x, offset_biases=offset_biases, effective_count=effective_count)
        
        # remove hallucination mask token
        if self.config.backbone.use_mask_token:
            x = x[:, :-1]

        return x, mask, ids_restore, ids_keep, ldmk_embs, pos_embed_masked, ldmks


    def apply_cls_token(self, x, pos_cls_embed, imp_mask_masked, full_importance_mask, rel_kp_embs=None):
        B = x.shape[0]
        cls_token = self.backbone.cls_token + pos_cls_embed
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        one_vec = torch.ones(B, 1, device=x.device, dtype=torch.bool)
        imp_mask_masked = torch.cat([one_vec, imp_mask_masked], dim=1)
        full_importance_mask = torch.cat([one_vec, full_importance_mask], dim=1)
        if self.use_rpe:
            _b, _, _k, _c = rel_kp_embs.shape
            dummy_kp_emb = torch.full((_b, 1, _k, _c), INVALID_KPRPE_EMB_VALUE, device=rel_kp_embs.device)
            rel_kp_embs = torch.cat([dummy_kp_emb, rel_kp_embs], dim=1)
        return x, imp_mask_masked, full_importance_mask, rel_kp_embs


    def apply_mask_token(self, x, imp_mask_masked, full_importance_mask, rel_kp_embs=None):
        B = x.shape[0]
        if self.config.backbone.use_mask_token:
            # replace unimportant patches with mask token
            to_replace = x[~imp_mask_masked]
            x[~imp_mask_masked] = self.mask_token.expand(to_replace.shape[0], -1)
            if self.use_rpe:
                _b, _, _k, _c = rel_kp_embs.shape
                dummy_kp_emb = torch.full((_b, 1, _k, _c), INVALID_KPRPE_EMB_VALUE, device=rel_kp_embs.device)
                rel_kp_embs[~imp_mask_masked] = INVALID_KPRPE_EMB_VALUE
                rel_kp_embs = torch.cat([rel_kp_embs, dummy_kp_emb], dim=1)

            # add hallucination mask token
            halluci_mask_token = self.mask_token.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat((x, halluci_mask_token), dim=1)
            n_missing_mask = (full_importance_mask.shape[1] - imp_mask_masked.sum(1)).unsqueeze(1)
            n_missing_mask = torch.clip(n_missing_mask, min=1)
            one_vec = torch.ones(B, x.shape[1]-1, device=x.device)
            effective_count = torch.cat([one_vec, n_missing_mask], dim=1)

        else:
            effective_count = None
        return x, effective_count, rel_kp_embs


    def forward(self, x, foreground_masks=None, ldmks=None, return_intermediate=False, **kwargs):
        backbone_config = self.config.backbone
        out_backbone = self.mrm_forward(x, foreground_masks, ldmks, subset_method=self.config.patch_embed_config.subset_method)
        feat, _, _, ids_keep, ldmk_embs, pos_embed_masked, used_ldmks = out_backbone

        record_reader_indices = kwargs.get('record_reader_indices', None)
        out_head = self.head(feat, x_mask_ids_keep=ids_keep, pos_embs=pos_embed_masked, 
                             keypoint_embs=ldmk_embs, return_intermediate=True, ldmks=used_ldmks,
                             record_reader_indices=record_reader_indices)
        if isinstance(out_head, tuple):
            final_feat, out_head = out_head
        else:
            final_feat = out_head

        if return_intermediate:
            return final_feat, out_backbone, out_head
        return final_feat


def make_relative_keypoints_embed(ldmks, square_pos_embed, pos_embed_masked):
    B = ldmks.shape[0]
    ldmks_grid =(ldmks * 2 - 1).unsqueeze(2).float()
    ldmk_embs = F.grid_sample(square_pos_embed.expand(B, -1, -1, -1), ldmks_grid, align_corners=True)
    ldmk_embs = ldmk_embs.transpose(1, 3)
    ldmk_diff_embs = pos_embed_masked.unsqueeze(2) - ldmk_embs
    # set tokens created from invalid ldmks to -2
    invalid_ldmks = (ldmks == -1).all(dim=2)
    ldmk_diff_embs = ldmk_diff_embs.transpose(1,2)
    ldmk_diff_embs[invalid_ldmks] = INVALID_KPRPE_EMB_VALUE
    ldmk_diff_embs = ldmk_diff_embs.transpose(1,2)
    return ldmk_diff_embs