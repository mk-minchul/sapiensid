import torch

def create_square_head_bbox_from_keypoints_batched(keypoints, padding=0.1, min_points=3, aux_condition=None):
    """Creates square bounding boxes covering the given batched keypoints.

    Args:
        keypoints (torch.Tensor): Batched keypoints (B, 17, 2).
        padding (float): Padding around the box.

    Returns:
        torch.Tensor: Bounding boxes (B, 4).
        if condition is not met, element of bboxes is set to -1
    """

    # Create a mask for valid keypoints
    valid_mask = (keypoints != -1).all(dim=-1).unsqueeze(-1)  # Shape: (B, 17, 1)
    
    # Find min and max coordinates for each batch
    min_coords, _ = torch.where(valid_mask, keypoints, torch.tensor(float('inf'))).min(dim=1)
    max_coords, _ = torch.where(valid_mask, keypoints, torch.tensor(float('-inf'))).max(dim=1)
    
    # Calculate center and size of the bounding boxes
    centers = (min_coords + max_coords) / 2

    # if nose is valid, center = (center + nose) / 2
    nose = keypoints[:, 0]
    nose_is_valid = (nose != -1).all(dim=-1)
    centers[nose_is_valid] = (centers[nose_is_valid] + nose[nose_is_valid]) / 2

    # compute distance from center to farthest keypoint
    # sizes = torch.max(max_coords - min_coords, dim=-1).values
    keypoint_euc_distances = torch.norm(keypoints - centers.unsqueeze(1), dim=-1, p=2, keepdim=True)
    keypoint_euc_distances[~valid_mask] = -1
    max_distances, _ = keypoint_euc_distances.max(dim=1)
    sizes = (max_distances * 2).squeeze(1)
    
    # Add padding
    half_sizes = sizes.unsqueeze(1) * (1 + padding) / 2
    
    # Calculate square bounding box coordinates
    x1y1 = torch.clamp(centers - half_sizes, min=0)
    x2y2 = torch.clamp(centers + half_sizes, max=1)
    
    # Combine coordinates into a single tensor
    bboxes = torch.cat([x1y1, x2y2], dim=-1).float()
    if min_points is not None:
        usable = valid_mask.sum(1).squeeze(1) >= min_points
        bboxes[~usable] = -1
    
    if aux_condition is not None:
        # there has to be at least one point from aux_condition indices
        aux_valid_mask = valid_mask[:, aux_condition]
        usable = aux_valid_mask.sum(1).squeeze(1) >= 1
        bboxes[~usable] = -1
    
    return bboxes


def create_square_bbox_from_keypoints_batched(keypoints, padding=0.1, min_points=3, aux_condition=None):
    """Creates square bounding boxes covering the given batched keypoints.

    Args:
        keypoints (torch.Tensor): Batched keypoints (B, 17, 2).
        padding (float): Padding around the box.

    Returns:
        torch.Tensor: Bounding boxes (B, 4).
        if condition is not met, element of bboxes is set to -1
    """

    # Create a mask for valid keypoints
    valid_mask = (keypoints != -1).all(dim=-1).unsqueeze(-1)  # Shape: (B, 17, 1)
    
    # Find min and max coordinates for each batch
    min_coords, _ = torch.where(valid_mask, keypoints, torch.tensor(float('inf'))).min(dim=1)
    max_coords, _ = torch.where(valid_mask, keypoints, torch.tensor(float('-inf'))).max(dim=1)
    
    # Calculate center and size of the bounding boxes
    centers = (min_coords + max_coords) / 2
    sizes = torch.max(max_coords - min_coords, dim=-1).values
    
    # Add padding
    half_sizes = sizes.unsqueeze(1) * (1 + padding) / 2
    
    # Calculate square bounding box coordinates
    x1y1 = torch.clamp(centers - half_sizes, min=0)
    x2y2 = torch.clamp(centers + half_sizes, max=1)
    
    # Combine coordinates into a single tensor
    bboxes = torch.cat([x1y1, x2y2], dim=-1).float()
    if min_points is not None:
        usable = valid_mask.sum(1).squeeze(1) >= min_points
        bboxes[~usable] = -1
    
    if aux_condition is not None:
        # there has to be at least one point from aux_condition indices
        aux_valid_mask = valid_mask[:, aux_condition]
        usable = aux_valid_mask.sum(1).squeeze(1) >= 1
        bboxes[~usable] = -1
    
    return bboxes


def augment_bboxes(bboxes, shift_range=0.1, scale_range=0.1):
    """
    Augment bounding boxes by shifting the center and changing the scale.
    
    :param bboxes: tensor of shape (B, 4) with normalized coordinates [x1, y1, x2, y2]
    :param shift_range: maximum shift as a fraction of bbox width/height
    :param scale_range: maximum scale change
    :return: augmented bboxes tensor of shape (B, 4)
    """
    device = bboxes.device
    B = bboxes.shape[0]
    
    # Calculate current centers and sizes
    centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2
    sizes = bboxes[:, 2:] - bboxes[:, :2]
    
    # Generate random shifts
    shifts = (torch.rand(B, 2, device=device) * 2 - 1) * shift_range * sizes
    
    # Generate random scales
    scales = 1 + (torch.rand(B, 2, device=device) * 2 - 1) * scale_range
    
    # Apply shifts and scales
    new_centers = centers + shifts
    new_sizes = sizes * scales
    
    # Calculate new bounding box coordinates
    new_bboxes = torch.cat([new_centers - new_sizes / 2, new_centers + new_sizes / 2], dim=1)
    
    # Clip to ensure values are between 0 and 1
    new_bboxes = torch.clamp(new_bboxes, 0, 1)
    
    return new_bboxes

