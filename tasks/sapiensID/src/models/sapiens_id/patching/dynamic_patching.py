import numpy as np
import torch
import torch.nn.functional as F
import copy


def to_abs_bbox(rel_bbox, img_size):
    # bbox: tensor of shape (4)
    # img_size: tuple of (H, W)
    # return tensor of shape (4)
    H, W = img_size
    scale = torch.tensor([W, H, W, H], device=rel_bbox.device)
    abs_bbox = rel_bbox * scale

    # round and convert to int
    abs_bbox = abs_bbox.round().int()
    return abs_bbox


def make_importance_map_batch(crop_size, patch_size, patch_info_batch):
    cumulative_importance_maps = []
    for patch_info_list in patch_info_batch:
        cumulative_importance_map, _ = make_importance_map(crop_size, patch_size, patch_info_list)
        cumulative_importance_map = cumulative_importance_map.unsqueeze(0).unsqueeze(0).float()
        cumulative_importance_maps.append(cumulative_importance_map)
    return cumulative_importance_maps


def make_importance_map(crop_size, patch_size, patch_info_list):
    """
    Generates an importance map for an image based on given patch information.
    """
    H_img, W_img = crop_size
    base_patch_size = patch_size

    # Compute the number of base patches along each dimension
    H_base = H_img // base_patch_size
    W_base = W_img // base_patch_size

    device = patch_info_list[0]["bbox"].device

    # Compute base patch grid coordinates
    x_base_coords = torch.arange(W_base, device=device) * base_patch_size
    y_base_coords = torch.arange(H_base, device=device) * base_patch_size
    X_base_grid, Y_base_grid = torch.meshgrid(x_base_coords, y_base_coords, indexing='ij')
    X_base_grid = X_base_grid.t()
    Y_base_grid = Y_base_grid.t()

    # Initialize the importance map over base patches
    cumulative_importance_map = torch.zeros((H_base, W_base), dtype=torch.int32, device=device)

    # Update importance map based on bounding boxes
    current_importance_maps = []
    for patch_info in patch_info_list:
        bbox = patch_info["bbox"]
        importance = patch_info["importance"]
        x1_norm, y1_norm, x2_norm, y2_norm = bbox

        x1_pixel = x1_norm * W_img
        y1_pixel = y1_norm * H_img
        x2_pixel = x2_norm * W_img
        y2_pixel = y2_norm * H_img

        x_overlap = (X_base_grid + base_patch_size > x1_pixel) & (X_base_grid < x2_pixel)
        y_overlap = (Y_base_grid + base_patch_size > y1_pixel) & (Y_base_grid < y2_pixel)
        mask = x_overlap & y_overlap

        current_importance_map = torch.zeros_like(cumulative_importance_map)
        current_importance_map = torch.where(
            mask,
            torch.tensor(importance, dtype=torch.int32),
            current_importance_map
        )
        current_importance_maps.append(current_importance_map)
        cumulative_importance_map = torch.where(
            mask,
            torch.maximum(cumulative_importance_map, torch.tensor(importance, dtype=torch.int32)),
            cumulative_importance_map
        )
    return cumulative_importance_map, current_importance_maps


def make_dynamic_patch_v2(images, face_bboxes, body_bboxes, foreground_masks, base_patch_size=32):

    B, C, H, W = images.shape
    patch_info_batch = []
    whole_bbox = torch.tensor([0.0, 0.0, 1.0, 1.0], device=images.device)
    for body_bbox, face_bbox in zip(body_bboxes, face_bboxes):
        patch_info_list = [
            {"bbox": whole_bbox, "importance": 0, "patch_size": base_patch_size, "crop_size": (384, 384)},
            {"bbox": body_bbox, "importance": 1, "patch_size": base_patch_size, "crop_size": (384, 384)},
            {"bbox": face_bbox, "importance": 2, "patch_size": base_patch_size, "crop_size": (384, 384)},
        ]
        patch_info_batch.append(patch_info_list)
    
    # make importance map
    cumulative_importance_maps = make_importance_map_batch((H, W), 1, patch_info_batch)

    # crop imporance map
    cumulative_importance_maps = torch.cat(cumulative_importance_maps, dim=0)

    # make patches of imaage for each importance level
    patches_dict = {}
    importance_list = [patch_info['importance'] for patch_info in patch_info_batch[0]][::-1]
    cropsize_list = [patch_info['crop_size'] for patch_info in patch_info_batch[0]][::-1]
    patchsize_list = [patch_info['patch_size'] for patch_info in patch_info_batch[0]][::-1]
    for importance, crop_size, patch_size in zip(importance_list, cropsize_list, patchsize_list):
        unfold_fn = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
        bboxes = torch.stack([patch_info[importance]['bbox'] for patch_info in patch_info_batch], dim=0)
        grid = create_sampling_grid(bboxes, crop_size[0], crop_size[1])

        cropped_img = F.grid_sample(images, grid, align_corners=False)
        patches_img = unfold_fn(cropped_img).transpose(1, 2).contiguous()  # Shape: (N, N_patches, C * patch_size^2)
        patches_img = patches_img.view(B, -1, C, patch_size, patch_size)  # Shape: (N, N_patches, C, patch_size, patch_size)

        cropped_imap = F.grid_sample(cumulative_importance_maps, grid, align_corners=False, mode='nearest')
        patches_imap = unfold_fn(cropped_imap).transpose(1, 2).contiguous()
        # important_mask = (patches_imap == importance).all(dim=2)
        important_mask = (patches_imap == importance).float().mean(dim=2) > 0.5
        if importance == 0:
            important_mask = important_mask & foreground_masks
        patches_dict[importance] = {'patches': patches_img, 
                                    'important_mask': important_mask,
                                    'crop_size': crop_size,
                                    'patch_size': patch_size}
    
    # post process temporarily
    result = []
    for i in range(B):
        info = {}
        for importance in importance_list:
            info[importance] = {'patches': patches_dict[importance]['patches'][i].unsqueeze(0),
                                'important_mask': patches_dict[importance]['important_mask'][i],
                                'crop_size': patches_dict[importance]['crop_size'],
                                'patch_size': patches_dict[importance]['patch_size']}
        result.append(info)
    return result


def make_dynamic_patch_v1(images, face_bboxes, body_bboxes, foreground_masks, base_patch_size=32):

    B, C, H, W = images.shape
    patch_info_batch = []
    whole_bbox = torch.tensor([0.0, 0.0, 1.0, 1.0], device=images.device)
    for body_bbox, face_bbox in zip(body_bboxes, face_bboxes):
        patch_info_list = [
            {"bbox": whole_bbox, "importance": 0, "patch_size": base_patch_size, "crop_size": (384, 384)},
            {"bbox": body_bbox, "importance": 1, "patch_size": base_patch_size, "crop_size": (384, 384)},
            {"bbox": face_bbox, "importance": 2, "patch_size": base_patch_size, "crop_size": (384, 384)},
        ]
        patch_info_batch.append(patch_info_list)
    
    # make importance map
    cumulative_importance_maps = make_importance_map_batch((H, W), 1, patch_info_batch)

    result = []
    for image, foreground_mask, patch_info_list, cumulative_importance_map in zip(images, foreground_masks, patch_info_batch, cumulative_importance_maps):
        N = 1
        image = image.unsqueeze(0)  # Shape: (1, C, H, W)
        assert cumulative_importance_map.shape[2:] == image.shape[2:], "Importance map and image must have the same shape"

        # Unfold the image into patches
        patches_dict = {}
        for patch_info in patch_info_list[::-1]:
            if patch_info['bbox'].min() == -1:
                # bad bbox (create zero patch)
                npatch = (H // patch_info['patch_size']) * (W // patch_info['patch_size'])
                null_patch = torch.zeros(N, npatch, C, patch_info["patch_size"], patch_info["patch_size"], device=images.device)
                patches_dict[patch_info["importance"]] = {'patches': null_patch, 
                                                        'important_mask': torch.zeros(npatch, device=images.device, dtype=torch.bool),
                                                        'crop_size': patch_info["crop_size"],
                                                        'patch_size': patch_info["patch_size"]}
                continue

            abs_bbox = to_abs_bbox(patch_info["bbox"], (H, W))
            cropped = image[:, :, abs_bbox[1]:abs_bbox[3], abs_bbox[0]:abs_bbox[2]]
            resized = torch.nn.functional.interpolate(
                cropped, size=patch_info["crop_size"], mode="bilinear", align_corners=None)
            unfold_fn = torch.nn.Unfold(kernel_size=patch_info["patch_size"], stride=patch_info["patch_size"])
            patches = unfold_fn(resized).transpose(1, 2).contiguous()  # Shape: (N, N_patches, C * patch_size^2)
            patches = patches.view(N, -1, C, patch_info["patch_size"], patch_info["patch_size"])  # Shape: (N, N_patches, C, patch_size, patch_size)

            # find out which patches are not covered by the current importance map
            importance = patch_info["importance"]
            cropped_importance_map = cumulative_importance_map[:, :, abs_bbox[1]:abs_bbox[3], abs_bbox[0]:abs_bbox[2]]
            resized_importance_map = torch.nn.functional.interpolate(
                cropped_importance_map, size=patch_info["crop_size"], mode="nearest", align_corners=None)
            patches_importance = unfold_fn(resized_importance_map).transpose(1, 2).contiguous()
            important_mask = (patches_importance == importance).all(dim=2).squeeze(0)
            important_mask = (patches_importance == importance).float().mean(dim=2).squeeze(0) > 0.5

            if importance == 0:
                important_mask = important_mask & foreground_mask

            key = patch_info["importance"]
            patches_dict[key] = {'patches': patches, 
                                'important_mask': important_mask,
                                'crop_size': patch_info["crop_size"],
                                'patch_size': patch_info["patch_size"]}
        result.append(patches_dict)
    return result



def visualize_patches(dynamic_patches, save_name='vis.png', fade_non_important=True):
    alL_result = []
    for patches_dict in dynamic_patches:
        restored_images = {}

        for importance_level, data in patches_dict.items():
            patches = data['patches']  # Shape: (N, N_patches, C, patch_size, patch_size)
            important_mask = data['important_mask']  # Shape: (N_patches)
            crop_size = data['crop_size']  # (H, W)
            patch_size = data['patch_size']  # scalar

            N, N_patches, C, _, _ = patches.shape

            # Compute the number of patches along height and width
            H_patches = crop_size[0] // patch_size
            W_patches = crop_size[1] // patch_size

            # Verify that the number of patches matches
            assert H_patches * W_patches == N_patches, "Mismatch in number of patches"

            # Zero out the patches that are not important
            patches = patches.clone()
            not_important_mask = ~important_mask
            # patches[:, not_important_mask.cpu()] = 0
            if fade_non_important:
                faded = (patches[:, not_important_mask.cpu()] * 0.5 + 0.5) * 0.3
                patches[:, not_important_mask.cpu()] = (faded - 0.5) / 0.5

            border_color=(1.0, 0.0, 0.0)
            border_color_tensor = torch.tensor(border_color, device=patches.device, dtype=patches.dtype)
            border_color_tensor = border_color_tensor.view(1, 1, C, 1, 1)
            border_width = 1 + 2*importance_level
            # border_width = 1
            # Apply borders to the patches
            if border_width > 0:
                patches[:, :, :, :border_width, :] = border_color_tensor
                patches[:, :, :, -border_width:, :] = border_color_tensor
                patches[:, :, :, :, :border_width] = border_color_tensor
                patches[:, :, :, :, -border_width:] = border_color_tensor


            # Reshape and permute to reconstruct the image
            restored_image = patches.view(N, H_patches, W_patches, C, patch_size, patch_size)
            restored_image = restored_image.permute(0, 3, 1, 4, 2, 5)
            restored_image = restored_image.contiguous().view(
                N, C, H_patches * patch_size, W_patches * patch_size
            )
            restored_images[importance_level] = restored_image.squeeze(0)  # Remove batch dimension if N=1

        alL_result.append(restored_images)
    all_concat = []
    for result in alL_result:
        concat = torch.cat([result[0], result[1], result[2]], dim=2)
        all_concat.append(concat)
    from general_utils.img_utils import visualize
    vis = torch.cat(all_concat, dim=1).unsqueeze(0).cpu()
    visualize(vis).save(save_name)

    return vis


def make_dynamic_pos_embed(square_pos_embed, face_bboxes, body_bboxes, output_size):
    '''
    for dummy boxes [-1, -1, -1, -1], it eventually samples all zero pos embed.
    '''
    B = face_bboxes.shape[0]
    face_grid = create_sampling_grid(face_bboxes, output_size[0], output_size[1])
    body_grid = create_sampling_grid(body_bboxes, output_size[0], output_size[1])
    face_pos_embed = F.grid_sample(square_pos_embed.expand(B, -1, -1, -1), face_grid, align_corners=False)
    body_pos_embed = F.grid_sample(square_pos_embed.expand(B, -1, -1, -1), body_grid, align_corners=False)
    whole_pos_embed = torch.nn.functional.interpolate(square_pos_embed, size=output_size, mode='bilinear', align_corners=False)
    dynamic_pos_embed = {
        2: face_pos_embed,
        1: body_pos_embed,
        0: whole_pos_embed.repeat(face_pos_embed.shape[0],1,1,1)
    }
    return dynamic_pos_embed


def visualize_pos_embed(dynamic_pos_embed, patch_vis):
    import cv2
    import numpy as np
    from general_utils.img_utils import tensor_to_numpy

    face_pos_embed = dynamic_pos_embed[2]
    body_pos_embed = dynamic_pos_embed[1]
    whole_pos_embed = dynamic_pos_embed[0]
    # vis = visualize_patches(dynamic_patches)
    vis_pos_embed = torch.cat([whole_pos_embed, body_pos_embed, face_pos_embed], dim=3)
    vis_pos_embed = torch.cat(torch.split(vis_pos_embed, 1, 0), dim=2).squeeze(0).permute(1,2,0).cpu().detach()
    vis_pos_embed_np = vis_pos_embed.numpy().astype(np.uint8)
    cv2.imwrite('pos_embed.png', vis_pos_embed_np)
    vis_np = tensor_to_numpy(patch_vis[0])
    vis_pos_embed_np = cv2.resize(vis_pos_embed_np, (vis_np.shape[1], vis_np.shape[0]))
    vis_pos_embed_np[vis_np == 127.5] = 0
    cv2.imwrite('pos_embed_blocked_out.png', vis_pos_embed_np)


def visualize_pos_embed_v2(dynamic_patches, dynamic_pos_embed, save_name='vis_pos_embed.png'):
    unfolded_pos_embeds = []
    importance_list = [0,1,2]
    B = len(dynamic_pos_embed[0])
    C = 3
    for imp in importance_list:
        pos_embed = dynamic_pos_embed[imp]
        H, W = dynamic_patches[0][imp]['crop_size']
        base_patch_size = dynamic_patches[0][imp]['patch_size']
        pos_emed_resized = F.interpolate(pos_embed, size=(H,W), mode='nearest')
        unfold_fn = torch.nn.Unfold(kernel_size=base_patch_size, stride=base_patch_size)
        unfolded_pos_embed = unfold_fn(pos_emed_resized).transpose(1, 2).contiguous()
        unfolded_pos_embed = unfolded_pos_embed.view(B, -1, C, base_patch_size, base_patch_size)
        unfolded_pos_embeds.append(unfolded_pos_embed)
    unfolded_pos_embeds = torch.stack(unfolded_pos_embeds, dim=1)
    unfolded_pos_embeds = ((unfolded_pos_embeds / 255) - 0.5) / 0.5
    dynamic_patches_for_pos_embed = copy.deepcopy(dynamic_patches)
    importance_list = dynamic_pos_embed.keys()
    for i in range(B):
        for importance in importance_list:
            dynamic_patches_for_pos_embed[i][importance]['patches'] = unfolded_pos_embeds[i][importance].unsqueeze(0).cpu().detach()
            # all 1 mask
            # all_one = torch.ones_like(dynamic_patches_for_pos_embed[i][importance]['important_mask'], dtype=torch.bool).cpu().detach()
            # dynamic_patches_for_pos_embed[i][importance]['important_mask'] = all_one

    vis = visualize_patches(dynamic_patches_for_pos_embed, save_name)
    return vis

def create_sampling_grid(bboxes, height, width):
    # Convert bboxes from [x1, y1, x2, y2] to [cx, cy, w, h]
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]

    # Create normalized 2D grid
    x = torch.linspace(-1, 1, width, device=bboxes.device, dtype=torch.float32)
    y = torch.linspace(-1, 1, height, device=bboxes.device, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

    # Adjust grid based on bounding box dimensions
    x_grid = x_grid.unsqueeze(0) * w.view(-1, 1, 1) / 2 + cx.view(-1, 1, 1)
    y_grid = y_grid.unsqueeze(0) * h.view(-1, 1, 1) / 2 + cy.view(-1, 1, 1)

    # Map adjusted grids from [0, 1] to [-1, 1]
    x_grid = x_grid * 2 - 1
    y_grid = y_grid * 2 - 1

    # Stack x and y grids
    grid = torch.stack([x_grid, y_grid], dim=-1)

    return grid



def visualize_masked_patch(base_patch_size, x_masked, ids_restore, images, save_name='vis1.png'):
    from torch.nn import Fold
    from general_utils.img_utils import visualize
    B, C, H, W = images.shape
    patch_size = (base_patch_size, base_patch_size)
    mask_token = torch.ones(1, 1, x_masked.shape[2], device=images.device)
    mask_tokens = mask_token.repeat(x_masked.shape[0], ids_restore.shape[1] - x_masked.shape[1], 1) ## B x (L * mask_ratio) x Decoder_dim
    x_ = torch.cat([x_masked, mask_tokens], dim=1) ## B x L x Decoder_dim
    x_ = torch.gather(
        x_,
        dim=1,
        index=ids_restore.unsqueeze(-1).repeat(1, 1, x_masked.shape[2]))

    border_color=(1.0, 0.0, 0.0)
    border_color_tensor = torch.tensor(border_color, device=x_.device, dtype=x_.dtype)
    border_color_tensor = border_color_tensor.view(1, 1, C, 1, 1)
    # border_width = 1 + 2*importance_level
    border_width = 1
    patches = x_.view(B, 432, 3, base_patch_size, base_patch_size)
    # Apply borders to the patches
    if border_width > 0:
        patches[:, :, :, :border_width, :] = border_color_tensor
        patches[:, :, :, -border_width:, :] = border_color_tensor
        patches[:, :, :, :, :border_width] = border_color_tensor
        patches[:, :, :, :, -border_width:] = border_color_tensor
    x_ = patches.view(B, 432, -1)

    x1 = x_[:, :144]
    vis1 = Fold(output_size=(H, W), kernel_size=patch_size, stride=patch_size)(x1.transpose(1, 2))
    x2 = x_[:, 144:144+144]
    vis2 = Fold(output_size=(H, W), kernel_size=patch_size, stride=patch_size)(x2.transpose(1, 2))
    x3 = x_[:, 144+144:]
    vis3 = Fold(output_size=(H, W), kernel_size=patch_size, stride=patch_size)(x3.transpose(1, 2))
    vis = torch.cat([vis1, vis2, vis3], dim=3)
    vis = torch.cat(torch.split(vis, 1, 0), dim=2).detach().cpu()
    # vis_orig = torch.cat(torch.split(images, 1, 0), dim=2)
    # vis = torch.cat([vis_orig, vis], dim=3)
    visualize(vis).save(save_name)
    if len(x_masked) == 1:
        x_ = x_.squeeze(0)
        x_masked_patch = x_[~(x_ == 1).all(1)]
        x_masked_patch = x_masked_patch.view(-1, C, base_patch_size, base_patch_size)
        visualize(x_masked_patch.cpu(), 
                  nrows=7, ncols=len(x_masked_patch)//7, 
                  border=True,
                  pershape=(32,32)).save(save_name.replace('.png', '_patch.png'))
    return vis



def visualize_masked_patch_with_ids_keep(x_masked, ids_restore, ids_keep, base_patch_size, H, W, save_name='vis1.png'):
    from torch.nn import Fold
    from general_utils.img_utils import visualize
    D = x_masked.shape[2]
    B = x_masked.shape[0]
    mask_token = torch.ones(1, 1, D, device=x_masked.device)
    n_full_patch = ids_restore.shape[1]
    x_restored = mask_token.repeat(B, n_full_patch, 1)
    x_restored.scatter_(dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D), src=x_masked)
    fold_fn = Fold(output_size=(H, W),
                   kernel_size=(base_patch_size, base_patch_size),
                   stride=(base_patch_size, base_patch_size))
    x_restored1 = x_restored[:, :144]
    x_restored2 = x_restored[:, 144:144+144]
    x_restored3 = x_restored[:, 144+144:]
    x_restored1 = fold_fn(x_restored1.transpose(1, 2))
    x_restored2 = fold_fn(x_restored2.transpose(1, 2))
    x_restored3 = fold_fn(x_restored3.transpose(1, 2))
    x_restored = torch.cat([x_restored1, x_restored2, x_restored3], dim=3)
    visualize(x_restored.cpu(), axis=0).save(save_name)

def visualize_patch_overlay(vis_patch, face_bboxes_in_body, body_bboxes):
    from general_utils.img_utils import visualize
    H, W = 384, 384
    C = 3
    B = len(face_bboxes_in_body)
    unfold_fn = torch.nn.Unfold(kernel_size=H, stride=H)
    vis = unfold_fn(vis_patch).transpose(1, 2).contiguous()
    vis = vis.view(B, 3, C, H, W)
    all_overlay = []
    for i in range(B):
        x1, x2, x3 = vis[i].split([1, 1, 1], dim=0)
        x1, x2, x3 = x1.clone(), x2.clone(), x3.clone()
        # warp face into bbox crop
        face_bbox = face_bboxes_in_body[i] * torch.tensor([W, H, W, H], device=face_bboxes_in_body.device)
        to_copy_region = x2[:, :, int(face_bbox[1]):int(face_bbox[3]), int(face_bbox[0]):int(face_bbox[2])]
        h, w = to_copy_region.shape[2], to_copy_region.shape[3]
        if h > 0 and w > 0:
            from_copy = torch.nn.functional.interpolate(x3, size=(h, w), mode='bilinear', align_corners=False)
            x2[:, :, int(face_bbox[1]):int(face_bbox[3]), int(face_bbox[0]):int(face_bbox[2])] = from_copy

        # warp body into original image
        body_bbox = body_bboxes[i] * torch.tensor([W, H, W, H], device=body_bboxes.device)
        to_copy_region = x1[:, :, int(body_bbox[1]):int(body_bbox[3]), int(body_bbox[0]):int(body_bbox[2])]
        h, w = to_copy_region.shape[2], to_copy_region.shape[3]
        if h > 0 and w > 0:
            from_copy = torch.nn.functional.interpolate(x2, size=(h, w), mode='bilinear', align_corners=False)
            x1[:, :, int(body_bbox[1]):int(body_bbox[3]), int(body_bbox[0]):int(body_bbox[2])] = from_copy

        combined = torch.cat([x1, x2, x3], dim=3)
        all_overlay.append(combined)
    all_overlay = torch.cat(all_overlay, dim=2)
    return all_overlay