import torch
from models.sapeins_body.modules.dynamic_patching import create_square_bbox_from_keypoints_batched, create_sampling_grid
import torch.nn.functional as F
from general_utils.img_utils import stack_images, visualize, tensor_to_pil

def get_face_pred(unnorm_images_rgb, face_net, face_preprocessor, face_prior_box):
    face_input = (unnorm_images_rgb - 0.5) / 0.5
    face_input = face_preprocessor(face_input)
    face_input = face_input.flip(1)
    result = face_net(face_input, face_prior_box)
    anchor_bbox_pred, anchor_cls_pred, anchor_ldmk_pred, merged, _ = result
    bbox, cls, ldmk = torch.split(merged, [4, 2, 10], dim=1)
    face_score = torch.nn.Softmax(dim=-1)(cls)[:,1:]
    return face_score, bbox, ldmk


def predict_face_bboxes(unnorm_images_rgb, keypoints, face_net, face_preprocessor, face_prior_box):
    output_size = (160, 160)
    face_keypoints = keypoints[:, :5]
    head_bboxes = create_square_bbox_from_keypoints_batched(face_keypoints, padding=1.5, min_points=2)
    head_grid = create_sampling_grid(head_bboxes, output_size[0], output_size[1])
    head_images = F.grid_sample(unnorm_images_rgb, head_grid, align_corners=False)

    face_scores, face_bboxes, face_ldmks = get_face_pred(head_images, face_net, face_preprocessor, face_prior_box)
    face_bboxes = face_bboxes.clip_(0, 1)

    # visualize(((head_images -0.5) / 0.5).cpu(), face_ldmks.cpu()).save('/mckim/temp/head_images.png')

    face_bboxes_in_original = map_face_bboxes_to_original(head_bboxes, face_bboxes)
    face_bboxes_in_original = pad_square_bbox(face_bboxes_in_original, padding=0.05)
    face_bboxes_in_original[head_bboxes.min(dim=1)[0] == -1] = -1

    # face_grid = create_sampling_grid(face_bboxes_in_original, output_size[0], output_size[1])
    # face_images = F.grid_sample(unnorm_images_rgb, face_grid, align_corners=False)
    # visualize(((face_images -0.5) / 0.5).cpu()).save('/mckim/temp/face_images.png')
    return face_bboxes_in_original


def map_face_bboxes_to_original(head_bboxes, face_bboxes):
    """
    Map face bounding boxes from cropped image space to original image space.
    
    :param head_bboxes: tensor of shape (16, 4) with normalized coordinates [x1, y1, x2, y2]
    :param face_bboxes: tensor of shape (16, 4) with normalized coordinates [x1, y1, x2, y2] in cropped space
    :return: face_bboxes_original: tensor of shape (16, 4) with normalized coordinates in original image space
    """
    # Ensure inputs are on the same device
    device = head_bboxes.device
    face_bboxes = face_bboxes.to(device)

    # Calculate the width and height of head bounding boxes
    head_widths = head_bboxes[:, 2] - head_bboxes[:, 0]
    head_heights = head_bboxes[:, 3] - head_bboxes[:, 1]

    # Scale face bounding boxes to head bounding box dimensions
    face_bboxes_scaled = face_bboxes.clone()
    face_bboxes_scaled[:, [0, 2]] *= head_widths.unsqueeze(1)
    face_bboxes_scaled[:, [1, 3]] *= head_heights.unsqueeze(1)

    # Translate face bounding boxes to head bounding box position
    face_bboxes_original = face_bboxes_scaled.clone()
    face_bboxes_original[:, [0, 2]] += head_bboxes[:, 0].unsqueeze(1)
    face_bboxes_original[:, [1, 3]] += head_bboxes[:, 1].unsqueeze(1)

    # Clip the values to ensure they're between 0 and 1
    face_bboxes_original = torch.clamp(face_bboxes_original, 0, 1)

    return face_bboxes_original

def pad_square_bbox(bbox_tensor, padding=0.1):
    """
    Add padding to bounding boxes in relative 0-1 space and make them square.
    
    Args:
    bbox_tensor (torch.Tensor): Tensor of shape (B, 4) containing bounding boxes in format (x1, y1, x2, y2)
    padding=0.1 (float): Percentage of the larger dimension to add as padding
    
    Returns:
    torch.Tensor: Tensor of shape (B, 4) containing padded square bounding boxes
    """
    x1, y1, x2, y2 = bbox_tensor.unbind(dim=1)
    
    width = x2 - x1
    height = y2 - y1
    
    # Make the box square by using the larger dimension
    size = torch.max(width, height)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Add padding
    size_with_padding = size * (1 + 2 * padding)
    
    # Calculate new coordinates
    x1_new = center_x - size_with_padding / 2
    y1_new = center_y - size_with_padding / 2
    x2_new = center_x + size_with_padding / 2
    y2_new = center_y + size_with_padding / 2
    
    # Ensure the box stays within 0-1 range
    x1_new = torch.clamp(x1_new, 0, 1)
    y1_new = torch.clamp(y1_new, 0, 1)
    x2_new = torch.clamp(x2_new, 0, 1)
    y2_new = torch.clamp(y2_new, 0, 1)
    
    return torch.stack([x1_new, y1_new, x2_new, y2_new], dim=1)
