from .rpe_options import make_kprpe_shared
from .rpe_options import  make_kprpe_input
from .RPE.KPRPE import kprpe_shared
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def make_rpe(rpe_config, depth, num_heads, emb_dim):
    if rpe_config is None:
        return None
    else:
        name = rpe_config.name
        if name == 'KPRPE_shared':
            return KPRPE(rpe_config, depth, num_heads, emb_dim)
        else:
            raise NotImplementedError(f"Unknow RPE: {name}")
    

class KPRPE(nn.Module):
    def __init__(self, rpe_config, depth, num_heads, emb_dim):
        super(KPRPE, self).__init__()
        self.rpe_config = rpe_config
        self.depth = depth
        self.num_heads = num_heads
        self.emb_dim = emb_dim

        # keypoint linear
        self.keypoint_linear, self.num_buckets = make_kprpe_shared(rpe_config, depth, num_heads)
        self.max_height, self.max_width = 14, 14
        self.bucket_ids = None

        self.rpe_setting = kprpe_shared.get_rpe_config(
            ratio=self.rpe_config.ratio,
            method=self.rpe_config.method,
            mode=self.rpe_config.mode,
            shared_head=self.rpe_config.shared_head,
            skip=1,
            rpe_on=self.rpe_config.rpe_on,
        )['rpe_k']

    def forward(self, rel_kp_embs, ids_keep, aux_tokens='cls_first_mask_last', patched_img_size=(12,12)):
        extra_ctx = make_kprpe_input(rel_kp_embs, self.keypoint_linear, self.rpe_config,
                                        self.depth, self.num_heads, self.num_buckets)
        
        # make rp_bucket (indication of how far each relationship)
        device = rel_kp_embs.device
        B = rel_kp_embs.shape[0]
        if self.bucket_ids is None:
            abs_pos = get_absolute_positions(self.max_height, self.max_width, torch.int32, device)
            max_L = self.max_height * self.max_width
            pos1 = abs_pos.view((max_L, 1, 2))
            pos2 = abs_pos.view((1, max_L, 2))
            diff = pos1 - pos2
            self.bucket_ids = rp_2d_product(diff,
                                            alpha=self.rpe_setting['alpha'],
                                            beta=self.rpe_setting['beta'],
                                            gamma=self.rpe_setting['gamma'],
                                            dtype=torch.int32)

        with torch.no_grad():
            masked_bucket_ids = subset_buckets_with_ids_keep(self.bucket_ids, ids_keep, patched_img_size)
            if aux_tokens == 'cls_first_mask_last':
                masked_bucket_ids = torch.nn.functional.pad(masked_bucket_ids, (1, 1, 1, 1), value=self.num_buckets-1)
            elif aux_tokens == 'cls_first':
                masked_bucket_ids = torch.nn.functional.pad(masked_bucket_ids, (1, 0, 1, 0), value=self.num_buckets-1)

        for _extra_ctx in extra_ctx:
            _extra_ctx['rp_bucket'] = masked_bucket_ids

        # stack all layers's relative keypoints and rp_buckets to do it once
        rel_keypoints = torch.cat([_extra_ctx['rel_keypoints'] for _extra_ctx in extra_ctx], dim=0)
        rp_buckets = torch.cat([_extra_ctx['rp_bucket'] for _extra_ctx in extra_ctx], dim=0)

        bb, L_query, L_key = rp_buckets.shape
        bb, H, N, nb = rel_keypoints.shape
        offset = torch.arange(0, L_query * nb, nb, device=rel_keypoints.device)[None, :, None]
        ctx_rp_bucket_flatten = (rp_buckets + offset).flatten(1)
        ctx_rp_bucket_flatten = ctx_rp_bucket_flatten.unsqueeze(1).expand(-1, H, -1)
        offset_bias = torch.gather(rel_keypoints.flatten(2), 2, ctx_rp_bucket_flatten).view(bb, -1, L_query, L_key)

        offset_bias = torch.split(offset_bias, B, dim=0)
        return offset_bias

@torch.no_grad()
def get_absolute_positions(height, width, dtype, device):
    rows = torch.arange(height, dtype=dtype, device=device).view(
        height, 1).repeat(1, width)
    cols = torch.arange(width, dtype=dtype, device=device).view(
        1, width).repeat(height, 1)
    return torch.stack([rows, cols], 2)

def rp_2d_product(diff, **kwargs):
    # convert beta to an integer since beta is a float number.
    beta_int = int(kwargs['beta'])
    S = 2 * beta_int + 1
    # the output of piecewise index function is in [-beta_int, beta_int]
    r = piecewise_index(diff[:, :, 0], **kwargs) + \
        beta_int  # [0, 2 * beta_int]
    c = piecewise_index(diff[:, :, 1], **kwargs) + \
        beta_int  # [0, 2 * beta_int]
    pid = r * S + c
    return pid

def piecewise_index(relative_position, alpha, beta, gamma, dtype):
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (alpha +
                                   torch.log(rp_abs_out / alpha) /
                                   math.log(gamma / alpha) *
                                   (beta - alpha)).round().clip(max=beta)).to(dtype)

    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        # round(x) when |x| <= alpha
        idx = idx.round().to(dtype)

    # assign the value when |x| > alpha
    idx[not_mask] = y_out
    return idx

def subset_buckets_with_ids_keep(buckets_ids, ids_keep, patched_img_size) -> torch.Tensor:
    h, w = patched_img_size
    assert h == w

    # find out what is the abs coordinate of ids_keep
    B = ids_keep.shape[0]
    abs_pos_2d = get_absolute_positions(h, w, torch.int32, buckets_ids.device).permute(2, 0, 1).unsqueeze(0)
    rel_pos_2d = abs_pos_2d / (h-1)
    full_pos_embed = [rel_pos_2d.flatten(2).permute(0, 2, 1) for importance in [0, 1, 2]]
    full_pos_embed = torch.cat(full_pos_embed, dim=1).expand(ids_keep.shape[0], -1, -1)
    pos_embed_masked = torch.gather(full_pos_embed, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, full_pos_embed.shape[2]))

    N1, N2 = buckets_ids.shape
    side = int(math.sqrt(N1))
    batched_buckets_ids = buckets_ids.unsqueeze(0).expand(ids_keep.shape[0], -1, -1)
    batched_buckets_ids = batched_buckets_ids.float()

    # first subset the height dim
    ldmks_grid = (pos_embed_masked * 2 - 1).unsqueeze(-2)
    one_side_batched = batched_buckets_ids.view(B, side, side, N2).permute(0, 3, 1, 2)
    one_side_masked = F.grid_sample(one_side_batched, ldmks_grid, align_corners=True, mode='nearest').squeeze(-1)
    one_side_masked = one_side_masked.permute(0, 2, 1)

    # second subset the width dim
    both_side_batched = one_side_masked.view(B, -1, side, side)
    buckets_ids_masked = F.grid_sample(both_side_batched, ldmks_grid, align_corners=True, mode='nearest').squeeze(-1)

    return buckets_ids_masked.long()