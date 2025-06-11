import torch
import numpy as np
from .kp_helper import create_square_head_bbox_from_keypoints_batched, create_square_bbox_from_keypoints_batched, augment_bboxes

class KP19Preprocessor():

    def __init__(self, config):
        self.config = config
        self.kp_preproc_config = config.kp_preprocessor

    def __call__(self, keypoints, training):
        assert keypoints.size(1) == 22
        # from yolo
        # 0: Nose
        # 1-2: Left Eye, Right Eye
        # 3-4: Left Ear, Right Ear
        # 5-6: Left Shoulder, Right Shoulder
        # 7-8: Left Elbow, Right Elbow
        # 9-10: Left Wrist, Right Wrist
        # 11-12: Left Hip, Right Hip
        # 13-14: Left Knee, Right Knee
        # 15-16: Left Ankle, Right Ankle
        # from face5
        # 17-18: Left Eye, Right Eye
        # 19 : Nose
        # 20-21: Left Mouth, Right Mouth

        # yolo_kps = keypoints[:, 0:17]
        # face5_kps = keypoints[:, 17:22]

        face7_kps = keypoints[:, [19, 17, 18, 20, 21, 3, 4]]
        body_kps = keypoints[:, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
        combined_kps = torch.cat([face7_kps, body_kps], dim=1)

        if self.kp_preproc_config.drop_idv_ldmk_prob > 0 and training:
            # combined_kps: B x 19 x 2
            # make mask of size B x 19 and drop
            B = combined_kps.size(0)
            mask = torch.rand(B, 19, device=keypoints.device) > self.kp_preproc_config.drop_idv_ldmk_prob
            mask = mask.unsqueeze(-1).expand_as(combined_kps)
            combined_kps = torch.where(mask, combined_kps, torch.tensor(-1.0, device=combined_kps.device))

        face5_idx = [0, 1, 2, 3, 4]
        upper_torso_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        if self.kp_preproc_config.drop_part_ldmk_prob > 0 and training:
            if np.random.rand() < self.kp_preproc_config.drop_part_ldmk_prob:
                drop_type = np.random.choice(['face', 'body'], size=1, p=[0.5, 0.5])[0]
                if drop_type == 'face':
                    combined_kps[:, face5_idx] = -1
                elif drop_type == 'body':
                    combined_kps[:, upper_torso_idx] = -1

        face5_kps = combined_kps[:, face5_idx]
        upper_torso_kps = combined_kps[:, upper_torso_idx]

        # create bboxes
        face_bboxes = create_square_head_bbox_from_keypoints_batched(face5_kps, padding=0.3, min_points=2)
        body_bboxes = create_square_bbox_from_keypoints_batched(upper_torso_kps, padding=0.3, min_points=3, aux_condition=[7,8])

        # augment bboxes
        if training and self.kp_preproc_config.bbox_aug_shift_range > 0 and self.kp_preproc_config.bbox_aug_scale_range > 0:
            shift_range = self.kp_preproc_config.bbox_aug_shift_range
            scale_range = self.kp_preproc_config.bbox_aug_scale_range
            face_bboxes = augment_bboxes(face_bboxes, shift_range=shift_range, scale_range=scale_range)
            body_bboxes = augment_bboxes(body_bboxes, shift_range=shift_range, scale_range=scale_range)

        # landmark drop augmentation
        if self.kp_preproc_config.bbox_drop_prob > 0 and training:
            drop_indices = torch.randperm(B)[:int(B*self.kp_preproc_config.bbox_drop_prob)]
            drop_type = np.random.choice(['face', 'body', 'both'], size=1, p=[0.4, 0.4, 0.2])[0]
            if drop_type == 'face':
                face_bboxes[drop_indices] = -1
            elif drop_type == 'body':
                body_bboxes[drop_indices] = -1
            else:
                face_bboxes[drop_indices] = -1
                body_bboxes[drop_indices] = -1

        return combined_kps, body_bboxes, face_bboxes
