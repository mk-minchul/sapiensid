import torch
import math

def unfold_mask(foreground_masks, patch_size, image_size):
    H, W = image_size
    foreground_masks = foreground_masks.unsqueeze(1)
    _, _, mH, mW = foreground_masks.shape
    patch_size_mask = (patch_size[0] / H * mH, patch_size[1] / W * mW)
    assert patch_size_mask[0] % 1 == 0
    assert patch_size_mask[1] % 1 == 0
    patch_size_mask = (int(patch_size_mask[0]), int(patch_size_mask[1]))
    unfold_fn = torch.nn.Unfold(kernel_size=patch_size_mask, stride=patch_size_mask)
    unfolded_foreground_masks = unfold_fn(foreground_masks.float()).transpose(1, 2).contiguous()
    unfolded_foreground_masks = (unfolded_foreground_masks == 1).float().mean(dim=2).squeeze(0) > 0.5
    return unfolded_foreground_masks


def masking_based_on_fg(x, foreground_masks, len_keep, mask_offset=None):
    B, L, D = x.shape
    bg_mask = 1 - foreground_masks.float()
    bg_mask = bg_mask + torch.rand_like(bg_mask)*0.001

    if mask_offset is not None:
        bg_mask = bg_mask + mask_offset

    ids_shuffle = torch.argsort(bg_mask, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([B, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore, ids_keep



def get_square_pos_embed(backbone):
    pos_cls_embed = backbone.pos_embed[:, :1, :]
    pos_embed = backbone.pos_embed[:, 1:, :]
    n = pos_embed.size(1)
    ph, hw = int(math.sqrt(n)), int(math.sqrt(n))
    pos_embed = pos_embed.permute(0, 2, 1).view(1, -1, ph, hw)
    return pos_cls_embed, pos_embed


def masked_sampling(dynamic_patches, dynamic_pos_embed, mask_ratio, is_train, subset_method=None, ldmks=None):

    full_patches = [torch.cat([dynamic_patch[importance]['patches'] for importance in [0, 1, 2]], dim=1) for dynamic_patch in dynamic_patches]
    full_patches = torch.cat(full_patches, dim=0).flatten(2)

    full_importance_mask = [torch.cat([dynamic_patch[importance]['important_mask'] for importance in [0, 1, 2]], dim=0) for dynamic_patch in dynamic_patches]
    full_importance_mask = torch.stack(full_importance_mask, dim=0)

    if is_train:
        n_full = full_importance_mask.size(1)
        n_sample_token = int(n_full * (1-mask_ratio))
        if subset_method is not None:
            if 'face_first' in subset_method:
                prob = float(subset_method.split('_')[-1])
                one_length = full_patches.size(1) // 3
                has_body = (ldmks != -1).any(dim=2).sum(dim=1) >= 13
                use_face_masking = torch.rand(full_patches.size(0), device=full_patches.device) < prob
                use_face_masking = use_face_masking & has_body
                mask_offset = torch.cat([
                    torch.full((full_patches.size(0), one_length), -0.3, device=full_patches.device),
                    torch.full((full_patches.size(0), one_length), -0.2, device=full_patches.device),
                    torch.full((full_patches.size(0), one_length), -0.1, device=full_patches.device)
                ], dim=1)
                mask_offset[~use_face_masking] = 0
            else:
                raise ValueError(f'subset_method must be one of [face_first], got {subset_method}')
        else:
            mask_offset = None
    else:
        num_fg = full_importance_mask.sum(dim=1, keepdim=True)
        max_num_fg = num_fg.max()
        n_sample_token = max_num_fg
        # n_sample_token = full_patches.shape[1]
        mask_offset = None
    x_masked, mask, ids_restore, ids_keep = masking_based_on_fg(full_patches, full_importance_mask, n_sample_token, mask_offset)

    full_pos_embed = [dynamic_pos_embed[importance].flatten(2).permute(0, 2, 1) for importance in [0, 1, 2]]
    full_pos_embed = torch.cat(full_pos_embed, dim=1)
    pos_embed_masked = torch.gather(full_pos_embed, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, full_pos_embed.shape[2]))

    # for replacing any unimportant patch that is not masked out later
    imp_mask_masked = torch.gather(full_importance_mask, dim=1, index=ids_keep)

    return x_masked, mask, ids_restore, ids_keep, pos_embed_masked, imp_mask_masked, full_importance_mask


def snap_bbox_to_patch(bboxes, n_side):
    bboxes_scaled = bboxes * n_side
    x1, y1, x2, y2 = bboxes_scaled.unbind(dim=1)
    snapped_x1 = x1.floor()
    snapped_y1 = y1.floor()
    # snapped_x1 = x1.round()
    # snapped_y1 = y1.round()
    # w, h = (x2-x1).ceil(), (y2-y1).ceil()
    # snapped_x2 = snapped_x1 + w
    # snapped_y2 = snapped_y1 + h
    snapped_x2 = x2.ceil()
    snapped_y2 = y2.ceil()
    # snapped_x2 = x2.round()
    # snapped_y2 = y2.round()
    snapped_bboxes = torch.stack([snapped_x1, snapped_y1, snapped_x2, snapped_y2], dim=1)
    snapped_bboxes = snapped_bboxes / n_side
    snapped_bboxes = torch.clamp(snapped_bboxes, 0, 1)
    return snapped_bboxes

def warp_bbox_space_from_orig_to_body(face_bboxes_original, body_bboxes):
    """
    Map face bounding boxes from original image space to cropped head image space.
    
    :param body_bboxes: tensor of shape (B, 4) with normalized coordinates [x1, y1, x2, y2] in original image space
    :param face_bboxes_original: tensor of shape (B, 4) with normalized coordinates [x1, y1, x2, y2] in original image space
    :return: face_bboxes_head: tensor of shape (B, 4) with normalized coordinates in cropped head image space
    """
    # Ensure inputs are on the same device
    device = body_bboxes.device
    face_bboxes_original = face_bboxes_original.to(device)

    # Calculate the width and height of body bounding boxes
    body_widths = body_bboxes[:, 2] - body_bboxes[:, 0]
    body_heights = body_bboxes[:, 3] - body_bboxes[:, 1]

    # to prevent division by zero and warping to 0 space
    body_widths[body_widths==0] = 1
    body_heights[body_heights==0] = 1

    # Translate face bounding boxes relative to body bounding box position
    face_bboxes_translated = face_bboxes_original.clone()
    face_bboxes_translated[:, [0, 2]] -= body_bboxes[:, 0].unsqueeze(1)
    face_bboxes_translated[:, [1, 3]] -= body_bboxes[:, 1].unsqueeze(1)

    # Scale face bounding boxes to normalized body bounding box dimensions
    face_bboxes_body = face_bboxes_translated.clone()
    face_bboxes_body[:, [0, 2]] /= body_widths.unsqueeze(1)
    face_bboxes_body[:, [1, 3]] /= body_heights.unsqueeze(1)

    # Clip the values to ensure they're between 0 and 1
    face_bboxes_body = torch.clamp(face_bboxes_body, 0, 1)

    return face_bboxes_body

def warp_bbox_space_from_body_to_original(face_bboxes_body, body_bboxes):
    """
    Map face bounding boxes from cropped image space to original image space.
    
    :param body_bboxes: tensor of shape (16, 4) with normalized coordinates [x1, y1, x2, y2]
    :param face_bboxes: tensor of shape (16, 4) with normalized coordinates [x1, y1, x2, y2] in cropped space
    :return: face_bboxes_original: tensor of shape (16, 4) with normalized coordinates in original image space
    """
    # Ensure inputs are on the same device
    device = body_bboxes.device
    face_bboxes = face_bboxes_body.to(device)

    # Calculate the width and height of body bounding boxes
    body_widths = body_bboxes[:, 2] - body_bboxes[:, 0]
    body_heights = body_bboxes[:, 3] - body_bboxes[:, 1]

    # to prevent division by zero and warping to 0 space
    body_widths[body_widths==0] = 1
    body_heights[body_heights==0] = 1

    # Scale face bounding boxes to body bounding box dimensions
    face_bboxes_scaled = face_bboxes.clone()
    face_bboxes_scaled[:, [0, 2]] *= body_widths.unsqueeze(1)
    face_bboxes_scaled[:, [1, 3]] *= body_heights.unsqueeze(1)

    # Translate face bounding boxes to body bounding box position
    face_bboxes_original = face_bboxes_scaled.clone()
    face_bboxes_original[:, [0, 2]] += body_bboxes[:, 0].unsqueeze(1)
    face_bboxes_original[:, [1, 3]] += body_bboxes[:, 1].unsqueeze(1)

    # Clip the values to ensure they're between 0 and 1
    face_bboxes_original = torch.clamp(face_bboxes_original, 0, 1)

    return face_bboxes_original


def make_face_ref_template():
    template = torch.tensor([(56.02, 71.736), 
                                (38.29, 51.70), 
                                (73.53, 51.50), 
                                (18.29,61.70), 
                                (93.53,61.50)]) / 112.
    template[:, 0] = template[:, 0] - template[:, 0].mean()
    template[:, 1] = template[:, 1] - template[:, 1].mean()
    template[:, 0] = template[:, 0] * 1.2
    template[:, 1] = template[:, 1] * 1.2
    template[:, 0] = template[:, 0] + 0.5
    template[:, 1] = template[:, 1] + 0.5
    return template.float()


def add_level_emb(dynamic_pos_embed, level_embed):
    for i in range(len(dynamic_pos_embed)):
        _level_embed = level_embed[i]
        dynamic_pos_embed[i] = dynamic_pos_embed[i] + _level_embed[None, :, None, None]
    return dynamic_pos_embed
