import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys
sys.path.insert(0, str(root))
import numpy as np
np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_
np.float = np.float_

from .base import BaseAligner
from torchvision import transforms
import torch
import numpy as np
from ultralytics import YOLO
import os
import pandas as pd
from .openpose_batched import OpenPoseDetectorBatched
from .face_detector.huggingface_model_utils import load_model_by_repo_id
from .face_detector.dfa import get_face_kp_predictor
from .alignment_helper import batched_adjust_kps
from .crop_helper import batch_crop, create_square_head_bbox_from_keypoints_batched, warp_bbox_space_from_body_to_original, warp_ldmk_space_from_body_to_original


class KeypointPredictor(BaseAligner):


    def __init__(self, config):
        super(KeypointPredictor, self).__init__()
        self.config = config

        self.rgb_mean = torch.tensor(self.config.rgb_mean)[None, :, None, None]
        self.rgb_std = torch.tensor(self.config.rgb_std)[None, :, None, None]

        # load yolo model
        if self.config.use_yolo:
            self.yolo_net = YOLO(os.path.expanduser("~/.cache/yolov8n-pose.pt"))
        else:
            self.yolo_net = None

        # get openpose detector
        if self.config.use_openpose:
            self.open_pose_det = OpenPoseDetectorBatched()

            yolo_kp_index = {"Name": ['Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear', 'Left Shoulder',
                                      'Right Shoulder', 'Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist',
                                      'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle']}
            self.yolo_kp_index = pd.DataFrame(yolo_kp_index)
            self.yolo_kp_index.index.name = 'Index'
            openpose_kp_index = {"Name": ['Nose', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist',
                                          'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Right Hip',
                                          'Right Knee', 'Right Ankle', 'Left Hip', 'Left Knee', 'Left Ankle',
                        'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear']}
            self.openpose_kp_index = pd.DataFrame(openpose_kp_index)
            self.openpose_kp_index.index.name = 'Index'
            self.openpose_to_yolo_mapping = self.create_openpose_to_yolo_mapping()
            # {0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15, 14: 2, 15: 1, 16: 4, 17: 3}
        else:
            self.open_pose_det = None
            self.openpose_to_yolo_mapping = None
        
        if self.config.face_detector:
            self.faceness_threshold = self.config.faceness_threshold
            if self.config.face_detector.startswith('dfa'):
                if self.config.face_detector == 'dfa_mobilenetv4':
                    pretrain_path=f'{root}/tasks/sapiensID/src/aligners/keypoint_predictor/pretrained_models/aligners/dfa_mobilenetv4_medium/mobilenetv4_Final.pth'
                else:
                    raise NotImplementedError(f'face detector {self.config.face_detector} not implemented')
                self.face_predictor = get_face_kp_predictor(pretrain_path=pretrain_path)
            else:
                HF_TOKEN = os.environ['HF_TOKEN']
                assert HF_TOKEN
                self.face_predictor=load_model_by_repo_id(repo_id=self.config.face_detector,
                                                          save_path=os.path.expanduser(f'~/cache/{self.config.face_detector}'),
                                                          HF_TOKEN=HF_TOKEN).to('cuda').eval()
                print(f'Aligner {self.config.face_detector} loaded')

        if self.config.use_cache:
            self.cache = {}
        else:
            self.cache = None


    def create_openpose_to_yolo_mapping(self):
        mapping = {}
        for op_idx, op_name in self.openpose_kp_index['Name'].items():
            yolo_idx = self.yolo_kp_index.index[self.yolo_kp_index['Name'] == op_name].tolist()
            if yolo_idx:
                mapping[op_idx] = yolo_idx[0]
        return mapping


    def eval(self):
        if self.yolo_net:
            self.yolo_net.model.eval()
            self.yolo_net.training = False
        if self.open_pose_det:
            self.open_pose_det.eval()
            for param in self.open_pose_det.parameters():
                param.requires_grad = False

    @classmethod
    def from_config(cls, config):
        model = cls(config)
        model.eval()
        return model
    
    def infer_using_openpose(self, unnorm_images_rgb):
        pose_results = self.open_pose_det(unnorm_images_rgb)
        # draw_bodyposes(unnorm_images_rgb, pose_results)

        keypoints_np = self.open_pose_det.pose_results_to_keypoints_np(pose_results)
        keypoints_openpose = torch.from_numpy(keypoints_np).to(unnorm_images_rgb.device)
        keypoints_yolo = self.to_yolo_keypoints(keypoints_openpose)
        return keypoints_yolo
    
    def adjust_left_right_keypoints(self, keypoints):
        # Swap left and right if the subject is looking back
        # 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow
        # 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13:
        # Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle
        # left_indices = np.array([3, 5])
        # right_indices = np.array([4, 6])
        left_indices = np.array([1, 3, 5, 7, 9, 11, 13, 15])
        right_indices = np.array([2, 4, 6, 8, 10, 12, 14, 16])
        visible_kps = (keypoints != -1).any(dim=-1)

        for left_index, right_index in zip(left_indices, right_indices):
            # Only consider pairs where both keypoints are visible
            left_visible = visible_kps[:, left_index]  # Shape: [batch_size]
            right_visible = visible_kps[:, right_index]
            both_visible = left_visible & right_visible
            # Compute difference in x-coordinates
            diff_x = keypoints[:, left_index, 0] - keypoints[:, right_index, 0]  # Shape: [batch_size]

            # Determine where the right keypoint is to the left of the left keypoint
            need_to_swap = (diff_x < 0) & both_visible  # Shape: [batch_size]

            # Perform swap where needed
            swap_indices = need_to_swap.nonzero(as_tuple=False).squeeze()
            if swap_indices.numel() > 0:
                # print(f"Swapping {swap_indices.numel()} keypoints by comparing left right")
                # print('index', left_index, right_index, swap_indices)
                temp = keypoints[swap_indices, left_index, :].clone()
                keypoints[swap_indices, left_index, :] = keypoints[swap_indices, right_index, :]
                keypoints[swap_indices, right_index, :] = temp

            # Handle cases where keypoints are not both visible
            # Replace x-coordinates of not visible keypoints with a large positive value
            only_left_visible = left_visible & ~right_visible
            only_right_visible = right_visible & ~left_visible
            only_left_visible_bad = only_left_visible & (keypoints[:, left_index, 0] < 0.5)
            only_right_visible_bad = only_right_visible & (keypoints[:, right_index, 0] > 0.5)
            half_visible_bad = only_left_visible_bad | only_right_visible_bad
            # Perform swap where needed
            swap_indices = half_visible_bad.nonzero(as_tuple=False).squeeze()
            if swap_indices.numel() > 0:
                # print(f"Swapping {swap_indices.numel()} keypoints by only half visible to middle line")
                # print('index', left_index, right_index, swap_indices)
                temp = keypoints[swap_indices, left_index, :].clone()
                keypoints[swap_indices, left_index, :] = keypoints[swap_indices, right_index, :]
                keypoints[swap_indices, right_index, :] = temp

        return keypoints
    

    def remove_outliers(self, keypoints):
        # 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow
        # 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13:
        # Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle
        nose_idx = 0
        left_eye_idx = 1
        right_eye_idx = 2
        left_ear_idx = 3
        right_ear_idx = 4
        left_shoulder_idx = 5
        right_shoulder_idx = 6

        # Determine visibility of keypoints
        visible_kps = (keypoints != -1).any(dim=-1)  # Shape: [batch_size, num_keypoints]

        # Check if left/right shoulders and ears are all visible
        torso_visible = visible_kps[:, [left_shoulder_idx, right_shoulder_idx, left_ear_idx, right_ear_idx]].all(dim=1)  # Shape: [batch_size]
        has_torso_indices = torso_visible.nonzero(as_tuple=False).squeeze()

        if has_torso_indices.numel() > 0:
            # Get x-coordinates of shoulders and ears
            left_shoulder_x = keypoints[has_torso_indices, left_shoulder_idx, 0]
            right_shoulder_x = keypoints[has_torso_indices, right_shoulder_idx, 0]
            left_ear_x = keypoints[has_torso_indices, left_ear_idx, 0]
            right_ear_x = keypoints[has_torso_indices, right_ear_idx, 0]

            # Create x-coordinate bounds for shoulders and ears
            shoulder_min_x = torch.min(left_shoulder_x, right_shoulder_x)
            shoulder_max_x = torch.max(left_shoulder_x, right_shoulder_x)
            ear_min_x = torch.min(left_ear_x, right_ear_x)
            ear_max_x = torch.max(left_ear_x, right_ear_x)

            # Union of shoulder and ear x-coordinate bounds
            bound_min_x = torch.min(shoulder_min_x, ear_min_x)
            bound_max_x = torch.max(shoulder_max_x, ear_max_x)

            # Get x-coordinates of nose and eyes
            nose_x = keypoints[has_torso_indices, nose_idx, 0]
            left_eye_x = keypoints[has_torso_indices, left_eye_idx, 0]
            right_eye_x = keypoints[has_torso_indices, right_eye_idx, 0]

            # Determine visibility of nose and eyes
            nose_visible = visible_kps[has_torso_indices, nose_idx]
            left_eye_visible = visible_kps[has_torso_indices, left_eye_idx]
            right_eye_visible = visible_kps[has_torso_indices, right_eye_idx]

            # Check if nose and eyes are within the bounds
            nose_out_of_bounds = nose_visible & ((nose_x < bound_min_x) | (nose_x > bound_max_x))
            left_eye_out_of_bounds = left_eye_visible & ((left_eye_x < bound_min_x) | (left_eye_x > bound_max_x))
            right_eye_out_of_bounds = right_eye_visible & ((right_eye_x < bound_min_x) | (right_eye_x > bound_max_x))

            # Set out-of-bounds keypoints to (-1, -1)
            keypoints[has_torso_indices[nose_out_of_bounds], nose_idx, :] = -1
            keypoints[has_torso_indices[left_eye_out_of_bounds], left_eye_idx, :] = -1
            keypoints[has_torso_indices[right_eye_out_of_bounds], right_eye_idx, :] = -1

        return keypoints

    def infer_using_yolo(self, unnorm_images_rgb):
        results = self.yolo_net(unnorm_images_rgb, device=unnorm_images_rgb.device, verbose=False)
        keypoints = [r.keypoints.xyn for r in results]
        empty = torch.ones((17, 2), device=unnorm_images_rgb.device) * -1
        keypoints = [_keypoints[0] if len(_keypoints[0]) > 0 else empty for _keypoints in keypoints ]
        keypoints = torch.stack(keypoints, dim=0)
        keypoints[keypoints==0] = -1
        return keypoints


    def forward(self, x):
        # body_kp detect
        # if yes:
        #   crop face and predict face_kp
        # if no:
        #   whole image and predict face_kp
        #   if face score < 0.5: 
        #       face_kp: null
        # warp back to full image space

        x = x.to(self.device)
        if self.rgb_mean.device != x.device:
            self.rgb_mean = self.rgb_mean.to(x.device)
            self.rgb_std = self.rgb_std.to(x.device)
        unnorm_images_rgb = (x * self.rgb_std + self.rgb_mean)
        unnorm_images_rgb = torch.clip(unnorm_images_rgb, 0, 1)

        hash_key = hash_batch(unnorm_images_rgb)
        if self.cache is not None and hash_key in self.cache:
            keypoints = self.cache[hash_key].clone().to(x.device)
        else:
            keypoints = None
            if self.config.use_openpose:
                assert keypoints is None, 'openpose and yolo cannot be used together'
                keypoints = self.infer_using_openpose(unnorm_images_rgb)

            if self.config.use_yolo:
                assert keypoints is None, 'openpose and yolo cannot be used together'
                keypoints = self.infer_using_yolo(unnorm_images_rgb)
        
        # make safe keypoints
        keypoints = self.adjust_left_right_keypoints(keypoints)
        keypoints = self.remove_outliers(keypoints)
        
        if self.cache is not None and hash_key not in self.cache:
            self.cache[hash_key] = keypoints.cpu().detach()
        
        if self.config.face_detector:
            # create head bbox (whole image if no body kps)
            face_kps = keypoints[:, 0:5]
            head_bboxes_xyxyn = create_square_head_bbox_from_keypoints_batched(face_kps, padding=0.5, min_points=2)
            no_body_kps_idx = (keypoints == -1).all(dim=-1).all(dim=-1)
            full_bbox = torch.tensor([[0, 0, 1, 1.0]], device=head_bboxes_xyxyn.device)
            head_bboxes_xyxyn[no_body_kps_idx] = full_bbox.repeat(no_body_kps_idx.sum(), 1)

            # crop head
            head_x = batch_crop(x, head_bboxes_xyxyn, crop_size=(160, 160))
            head_kps = batched_adjust_kps(head_bboxes_xyxyn, keypoints)

            # predict face kps
            if self.config.face_detector.startswith('dfa'):
                face_scores, face_bboxs_in_head, face_ldmks5_in_head = self.face_predictor(head_x)
            else:
                _, face_ldmks5_in_head, _, face_scores, _, face_bboxs_in_head = self.face_predictor(head_x)
            face_bboxes_in_whole = warp_bbox_space_from_body_to_original(face_bboxs_in_head, head_bboxes_xyxyn)
            face_ldmks5_in_whole = warp_ldmk_space_from_body_to_original(face_ldmks5_in_head, head_bboxes_xyxyn)
            # make face_ldmks5_in_whole square

            # if face_scores < 0.5, set keypoints to -1
            face_scores = face_scores.squeeze(1)
            below_idx = face_scores < self.faceness_threshold
            face_ldmks5_in_whole[below_idx] = -1


            # visualize(x.cpu(), keypoints.cpu(), ncols=8, nrows=8).save(f'/mckim/temp/1_start.png')
            # visualize(head_x.cpu(), ncols=8, nrows=8).save(f'/mckim/temp/2_cropped.png')
            # visualize(head_x.cpu(), head_kps.cpu(), ncols=8, nrows=8).save(f'/mckim/temp/3_head_kps.png')
            # visualize(head_x.cpu(), head_kps.cpu(), 
            #           texts=[f'{s:.2f}' for s in face_scores.cpu()],
            #           bboxes_xyxyn=face_bboxs_in_head.cpu(), 
            #           ncols=8, nrows=8).save(f'/mckim/temp/4_head_kps_bbox.png')
            # visualize(head_x.cpu(), face_ldmks5_in_head.cpu(), 
            #           texts=[f'{s:.2f}' for s in face_scores.cpu()],
            #           bboxes_xyxyn=face_bboxs_in_head.cpu(), 
            #           ncols=8, nrows=8).save(f'/mckim/temp/5_cropped_face_kps_bbox.png')
            # visualize(x.cpu(), keypoints.cpu(), bboxes_xyxyn=face_bboxes_in_whole, ncols=8, nrows=8).save(f'/mckim/temp/6_whole.png')
            # visualize(x.cpu(), face_ldmks5_in_whole.cpu(), bboxes_xyxyn=face_bboxes_in_whole, ncols=8, nrows=8).save(f'/mckim/temp/6_whole_face.png')

        else:
            face_ldmks5_in_whole = None
            face_bboxes_in_whole = None

        # get foreground mask based on white space
        foreground_masks = get_foreground_mask_by_whitespace(unnorm_images_rgb)

        # join keypoints and face keypoints
        combined_keypoints = self.join_yolo_and_face_keypoints(keypoints, face_ldmks5_in_whole)
        return  combined_keypoints, foreground_masks


    def to_yolo_keypoints(self, keypoints_openpose):
        # Create mapping tensors
        openpose_indices = torch.tensor(list(self.openpose_to_yolo_mapping.keys()), device=keypoints_openpose.device)
        yolo_indices = torch.tensor(list(self.openpose_to_yolo_mapping.values()), device=keypoints_openpose.device)
        
        # Use advanced indexing to map keypoints
        B, _, _ = keypoints_openpose.shape
        keypoints_yolo = torch.full((B, 17, 2), -1.0, dtype=keypoints_openpose.dtype, device=keypoints_openpose.device)
        keypoints_yolo[:, yolo_indices] = keypoints_openpose[:, openpose_indices]
        return keypoints_yolo

    def make_train_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform

    def make_test_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform
    
    def join_yolo_and_face_keypoints(self, yolo_keypoints, face5_keypoints):
        return torch.cat([yolo_keypoints, face5_keypoints], dim=1)
        



def get_foreground_mask_by_whitespace(images, white_threshold_ratio=0.99):
    # find the padded region by assuming that the padded region is white and 
    # padded either horizontally or vertically

    batch_size, channels, H, W = images.shape
    summed_images = images.sum(dim=1)
    white_threshold = channels * white_threshold_ratio

    row_means = summed_images.mean(dim=2)
    col_means = summed_images.mean(dim=1)

    content_rows_mask = row_means < white_threshold
    content_cols_mask = col_means < white_threshold

    def get_content_indices(mask, dimension_size):
        first_indices = torch.argmax(mask.float(), dim=1)
        reversed_mask = torch.flip(mask, dims=[1])
        last_indices_from_end = torch.argmax(reversed_mask.float(), dim=1)
        last_indices = dimension_size - last_indices_from_end - 1

        has_content = mask.any(dim=1)
        first_indices = torch.where(has_content, first_indices, torch.tensor(0, device=images.device))
        last_indices = torch.where(has_content, last_indices, torch.tensor(dimension_size - 1, device=images.device))

        return first_indices, last_indices

    first_content_row, last_content_row = get_content_indices(content_rows_mask, H)
    first_content_col, last_content_col = get_content_indices(content_cols_mask, W)

    row_indices = torch.arange(H, device=images.device).view(1, H, 1).expand(batch_size, H, W)
    col_indices = torch.arange(W, device=images.device).view(1, 1, W).expand(batch_size, H, W)

    first_content_row = first_content_row.view(batch_size, 1, 1)
    last_content_row = last_content_row.view(batch_size, 1, 1)
    first_content_col = first_content_col.view(batch_size, 1, 1)
    last_content_col = last_content_col.view(batch_size, 1, 1)

    masks = (
        (row_indices >= first_content_row) & (row_indices <= last_content_row) &
        (col_indices >= first_content_col) & (col_indices <= last_content_col)
    ).to(dtype=torch.bool)

    return masks


def hash_batch(unnorm_images_rgb):
    long_image = (unnorm_images_rgb * 255).round()
    mean1 = long_image.mean(dim=(2), keepdim=True).flatten()
    mean2 = long_image.mean(dim=(3), keepdim=True).flatten()
    mean = (mean1 + mean2) / 2
    hashed_mean = hash(mean.to(torch.uint8).cpu().numpy().tobytes())
    return hashed_mean
