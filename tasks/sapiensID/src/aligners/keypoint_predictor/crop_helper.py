import torch
import torch.nn.functional as F

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



def batch_crop(images, bboxes, crop_size):
    grid = create_sampling_grid(bboxes, crop_size[0], crop_size[1])
    cropped_img = F.grid_sample(images, grid, align_corners=False)
    return cropped_img



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


def warp_ldmk_space_from_body_to_original(face_ldmks, body_bboxes):
    """
    Map face bounding boxes from cropped image space to original image space.
    
    :param body_bboxes: tensor of shape (16, 4) with normalized coordinates [x1, y1, x2, y2]
    :param face_bboxes: tensor of shape (16, 4) with normalized coordinates [x1, y1, x2, y2] in cropped space
    :return: face_bboxes_original: tensor of shape (16, 4) with normalized coordinates in original image space
    """
    # Ensure inputs are on the same device
    device = body_bboxes.device
    face_ldmks = face_ldmks.to(device)

    # Calculate the width and height of body bounding boxes
    body_widths = body_bboxes[:, 2] - body_bboxes[:, 0]
    body_heights = body_bboxes[:, 3] - body_bboxes[:, 1]

    # to prevent division by zero and warping to 0 space
    body_widths[body_widths==0] = 1
    body_heights[body_heights==0] = 1

    # Scale face bounding boxes to body bounding box dimensions
    face_ldmks_scaled = face_ldmks.clone()
    face_ldmks_scaled[:, :, 0] *= body_widths.unsqueeze(1)
    face_ldmks_scaled[:, :, 1] *= body_heights.unsqueeze(1)

    # Translate face bounding boxes to body bounding box position
    face_ldmks_original = face_ldmks_scaled.clone()
    face_ldmks_original[:, :, 0] += body_bboxes[:, 0].unsqueeze(1)
    face_ldmks_original[:, :, 1] += body_bboxes[:, 1].unsqueeze(1)

    # Clip the values to ensure they're between 0 and 1
    face_ldmks_original = torch.clamp(face_ldmks_original, 0, 1)

    return face_ldmks_original