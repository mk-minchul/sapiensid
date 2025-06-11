import torch
import torch.nn.functional as F


def batched_adjust_kps(head_bboxes_xyxyn, keypoints):
    """
    Adjust keypoints relative to their bounding boxes in a batched manner.

    Parameters:
    - head_bboxes_xyxyn (torch.Tensor): Tensor of shape [batch_size, 4] containing bounding boxes.
    - keypoints (torch.Tensor): Tensor of shape [batch_size, 17, 2] containing keypoints.

    Returns:
    - adjusted_kps (torch.Tensor): Tensor of adjusted keypoints with the same shape as input keypoints.
    """
    assert keypoints.shape[1] == 17
    assert head_bboxes_xyxyn.shape[1] == 4

    # Extract bbox coordinates
    x1 = head_bboxes_xyxyn[:, 0]  # [batch_size]
    y1 = head_bboxes_xyxyn[:, 1]
    x2 = head_bboxes_xyxyn[:, 2]
    y2 = head_bboxes_xyxyn[:, 3]

    # Identify keypoints to ignore
    ignore_idx = (keypoints == -1).all(dim=-1)  # [batch_size, 17]

    # Calculate width and height of the bbox
    width = (x2 - x1).unsqueeze(1)  # [batch_size, 1]
    height = (y2 - y1).unsqueeze(1)  # [batch_size, 1]

    # Adjust keypoints
    x1 = x1.unsqueeze(1)
    y1 = y1.unsqueeze(1)
    adjusted_kps = (keypoints - torch.stack((x1, y1), dim=2)) / torch.stack((width, height), dim=2)

    # Clip values to ensure they're in the range [0, 1]; set out-of-range values to -1
    adjusted_kps = torch.where(
        (adjusted_kps < 0) | (adjusted_kps > 1),
        adjusted_kps.new_tensor(-1.0),
        adjusted_kps
    )

    # Set ignored keypoints to -1
    adjusted_kps[ignore_idx.unsqueeze(-1).expand(-1, -1, 2)] = -1.0

    return adjusted_kps


def adjust_kps(bbox_xyxyn, kps):
    assert kps.shape[0] == 17
    assert bbox_xyxyn.shape[0] == 4
    # Extract bbox coordinates
    x1, y1, x2, y2 = bbox_xyxyn

    # index of -1
    ignore_idx = (kps == -1).all(dim=-1)
    
    # Calculate width and height of the bbox
    width = x2 - x1
    height = y2 - y1
    
    # Adjust keypoints
    adjusted_kps = (kps - torch.tensor([[x1, y1]], device=kps.device)) / torch.tensor([[width, height]], device=kps.device)
    
    # Clip values to ensure they're in the range [0, 1] by setting out-of-range values to -1
    adjusted_kps = torch.where(adjusted_kps > 1, torch.tensor([-1.0, -1.0], device=adjusted_kps.device), adjusted_kps)
    adjusted_kps = torch.where(adjusted_kps < 0, torch.tensor([-1.0, -1.0], device=adjusted_kps.device), adjusted_kps)

    # set ignored keypoints to -1
    adjusted_kps[ignore_idx] = -1
    
    return adjusted_kps


def adjust_kps_to_padding(cropped_kps, cropped_x):
    _, _, h, w = cropped_x.size()
    max_dim = max(h, w)
    
    # Calculate padding
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    
    # Convert keypoints to absolute coordinates
    kps_abs = cropped_kps * torch.tensor([w, h], device=cropped_kps.device)
    
    # Add padding offsets
    kps_padded = kps_abs + torch.tensor([pad_w, pad_h], device=cropped_kps.device)
    
    # Normalize to [0, 1] in the new padded space
    kps_padded_norm = kps_padded / max_dim
    
    return kps_padded_norm

def adjust_keypoints_to_alignment(cropped_kps, thetas):
    adjusted_kps = adjust_ldmks(cropped_kps, thetas)
    # anything out of range [0, 1] set to -1
    adjusted_kps = torch.where(adjusted_kps > 1, torch.tensor([-1.0, -1.0], device=adjusted_kps.device), adjusted_kps)
    adjusted_kps = torch.where(adjusted_kps < 0, torch.tensor([-1.0, -1.0], device=adjusted_kps.device), adjusted_kps)
    return adjusted_kps


def align_larger_crop(thetas, crop_size, save_size, x, orig_pred_ldmks):
    # instead of using orig x, use large crop aligned by landmarks
    new_thetas = adjust_thetas_for_larger_output(thetas, 112, crop_size)
    grid = F.affine_grid(new_thetas, torch.Size((len(thetas), 3, save_size, save_size)), align_corners=True)
    large_crop_x = F.grid_sample(x + 1, grid, align_corners=True) - 1  # +1, -1 for making padding pixel 0
    large_crop_ldmks = adjust_ldmks(orig_pred_ldmks.view(-1, 5, 2), new_thetas)
    return large_crop_x, large_crop_ldmks, new_thetas


def adjust_thetas_for_larger_output(thetas, current_output_size, new_output_size):
    assert thetas.ndim == 3 and thetas.shape[1:] == (2, 3), "Thetas should be of shape (N, 2, 3)"

    # Extract current height and width
    current_height = current_output_size
    current_width = current_output_size

    # Extract new height and width
    new_height = new_output_size
    new_width = new_output_size

    # Calculate scaling factors
    scale_x = new_width / current_width
    scale_y = new_height / current_height

    # Adjust the thetas
    adjusted_thetas = thetas.clone()
    adjusted_thetas[:, 0, 0] *= scale_x
    adjusted_thetas[:, 0, 1] *= scale_y
    adjusted_thetas[:, 1, 0] *= scale_x
    adjusted_thetas[:, 1, 1] *= scale_y

    # Adjust the translation component
    # adjusted_thetas[:, 0, 2] = (thetas[:, 0, 2] + 1) * scale_x - 1
    # adjusted_thetas[:, 1, 2] = (thetas[:, 1, 2] + 1) * scale_y - 1

    return adjusted_thetas

def adjust_ldmks(ldmks, thetas):
    inv_thetas = inv_matrix(thetas).to(ldmks.device).float()
    _ldmks = torch.cat([ldmks, torch.ones((ldmks.shape[0], ldmks.shape[1], 1)).to(ldmks.device)], dim=2)
    ldmk_aligned = (((_ldmks) * 2 - 1) @ inv_thetas.permute(0,2,1)) / 2 + 0.5
    return ldmk_aligned

def inv_matrix(theta):
    # torch batched version
    assert theta.ndim == 3
    a, b, t1 = theta[:, 0,0], theta[:, 0,1], theta[:, 0,2]
    c, d, t2 = theta[:, 1,0], theta[:, 1,1], theta[:, 1,2]
    det = a * d - b * c
    inv_det = 1.0 / det
    inv_mat = torch.stack([
        torch.stack([d * inv_det, -b * inv_det, (b * t2 - d * t1) * inv_det], dim=1),
        torch.stack([-c * inv_det, a * inv_det, (c * t1 - a * t2) * inv_det], dim=1)
    ], dim=1)
    return inv_mat


def pad_to_square(x, value=0):
    _, _, h, w = x.size()

    # Calculate the padding needed to make the height and width equal
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2

    pad_h_extra = (max_dim - h) % 2
    pad_w_extra = (max_dim - w) % 2

    # Apply padding: (left, right, top, bottom)
    padding = (pad_w, pad_w + pad_w_extra, pad_h, pad_h + pad_h_extra)
    x_padded = F.pad(x, padding, mode='constant', value=value)

    return x_padded


def crop_upper_body_from_body(x, crop_kps, margin=0.5):
    # 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow
    # 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13:
    # Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle
    _, _, height, width = x.shape
    assert x.shape[0] == 1
    assert crop_kps.shape[0] == 1
    assert crop_kps.shape[1] == 17

    # available_keypoints
    visible_kps = torch.nonzero((crop_kps[0] != -1).all(dim=-1)).squeeze(dim=-1)
    upper_body_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6], device=crop_kps.device, dtype=visible_kps.dtype)
    lower_body_indices = torch.tensor([5,6,7,8,9,10,11,12,13,14,15,16], device=crop_kps.device, dtype=visible_kps.dtype)

    has_lower_body = torch.isin(lower_body_indices, visible_kps).any()
    if not has_lower_body:
        # No lower body keypoints, return the original image
        bbox_xyxyn = torch.tensor([[0, 0, 1, 1]], device=crop_kps.device, dtype=crop_kps.dtype)
        return x, bbox_xyxyn

    shoulder_indices = torch.tensor([5, 6], device=crop_kps.device, dtype=visible_kps.dtype)
    has_shoulder = torch.isin(shoulder_indices, visible_kps).any()
    if not has_shoulder:
        # No shoulder keypoints, return the original image
        bbox_xyxyn = torch.tensor([[0, 0, 1, 1]], device=crop_kps.device, dtype=crop_kps.dtype)
        return x, bbox_xyxyn

    mask = torch.isin(upper_body_indices, visible_kps)
    valid_kps = torch.masked_select(upper_body_indices, mask)

    if len(valid_kps) < 2:
        # Not enough visible keypoints, return the original image
        bbox_xyxyn = torch.tensor([[0, 0, 1, 1]], device=crop_kps.device, dtype=crop_kps.dtype)
        return x, bbox_xyxyn

    # Nose, Eyes, Ears, Shoulders
    upper_body_kps = crop_kps[0, valid_kps, :]
    upper_body_kps = torch.clip(upper_body_kps, 0, 1)
    abs_kps = upper_body_kps * torch.tensor([[width, height]], device=crop_kps.device, dtype=crop_kps.dtype)

    # Calculate the bounding box for the upper body
    min_x = torch.min(abs_kps[:, 0]).int().item()
    max_x = torch.max(abs_kps[:, 0]).int().item()
    min_y = torch.min(abs_kps[:, 1]).int().item()
    max_y = torch.max(abs_kps[:, 1]).int().item()
    if min_x == max_x or min_y == max_y:
        # No bounding box area, return the original image
        bbox_xyxyn = torch.tensor([[0, 0, 1, 1]], device=crop_kps.device, dtype=crop_kps.dtype)
        return x, bbox_xyxyn

    # Enlarge the cropping area
    box_width = max_x - min_x
    box_height = max_y - min_y

    min_x = max(0, int(min_x - 0.75 * margin * box_width))
    max_x = min(width, int(max_x + 0.75 * margin * box_width))
    min_y = max(0, int(min_y - 3 * margin * box_height))
    max_y = min(height, int(max_y + margin * box_height))

    # has left_shoulder or right_shoulder then cut the image to the shoulder
    lr_shoulder_idx = torch.tensor([5,6], device=crop_kps.device, dtype=visible_kps.dtype)
    if torch.isin(lr_shoulder_idx, visible_kps).any():
        index_of_shoulder_in_valid_kps = torch.where(torch.isin(valid_kps, lr_shoulder_idx))[0]
        abs_lr_shoulder = abs_kps[index_of_shoulder_in_valid_kps]
        max_y = int(abs_lr_shoulder[:, 1].max().item())

    # has both left_ear or right_ear then cut the image to the ear
    lr_ear_idx = torch.tensor([3,4], device=crop_kps.device, dtype=visible_kps.dtype)
    if torch.isin(lr_ear_idx, visible_kps).all():
        index_of_left_ear_in_valid_kps = torch.where(valid_kps == 3)[0]
        index_of_right_ear_in_valid_kps = torch.where(valid_kps == 4)[0]
        abs_left_ear = abs_kps[index_of_left_ear_in_valid_kps]
        abs_right_ear = abs_kps[index_of_right_ear_in_valid_kps]
        min_x = int(abs_left_ear[0, 0].item())
        max_x = int(abs_right_ear[0, 0].max().item())
        if min_x > max_x:
            min_x, max_x = max_x, min_x
        ear_width = (max_x - min_x) // 2
        min_x = max(0, int(min_x - ear_width))
        max_x = min(width, int(max_x + ear_width))

    # Crop the image
    cropped_image = x[:, :, min_y:max_y, min_x:max_x]
    width, height = x.shape[3], x.shape[2]
    bbox_xyxyn = torch.tensor([min_x / width, min_y / height,
                               max_x / width, max_y / height], device=crop_kps.device, dtype=crop_kps.dtype)
    bbox_xyxyn = bbox_xyxyn.unsqueeze(dim=0)
    return cropped_image, bbox_xyxyn

