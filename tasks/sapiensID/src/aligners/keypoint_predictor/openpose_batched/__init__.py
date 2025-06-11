from .open_pose import OpenposeDetector
from .open_pose import PoseResult, BodyResult, Keypoint
from .open_pose.img_utils import HWC3, resize_image
from .vis_helper import draw_bodyposes
import time
import numpy as np
import torch
from .result_formater import format_body_result, make_pose_result, select_best_pose
from .batch_helper import body_call_batch
import torch.nn as nn

def infer_batch(unnorm_images_rgb, body_model):
    '''
    unnorm_images_rgb: tensor[B, 3, H, W] in range [0.0, 1.0]
    body_model: nn.Module
    '''
    if unnorm_images_rgb.shape[2] == unnorm_images_rgb.shape[3]:
        input_images = torch.nn.functional.interpolate(unnorm_images_rgb, 
                                                    size=(512, 512), 
                                                    mode='bilinear', 
                                                    align_corners=False)
    else:
        # it doesn't need to be squared. but not tested yet. 
        # to make it work, just keep aspect ratio and resize to max(H, W) = 512
        raise NotImplementedError("Only support square image")

    input_images_bgr = input_images.flip(1)
    candidates, subsets = body_call_batch(body_model, input_images_bgr)
    all_results = []
    for candidate, subset in zip(candidates, subsets):
        bodies = format_body_result(candidate, subset)
        results = make_pose_result(bodies, input_images_bgr.shape[-2:], include_hand=False, include_face=False)
        best_pose = select_best_pose(results)
        all_results.append(best_pose)

    return all_results

class OpenPoseDetectorBatched(nn.Module):
    def __init__(self):
        super().__init__()
        open_pose_det = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        self.body_model = open_pose_det.body_estimation.model
        self.body_model.eval()

    def forward(self, unnorm_images_rgb):
        return infer_batch(unnorm_images_rgb, self.body_model)
    
    def kp_to_np(self, kp):
        return np.array([kp.x, kp.y])

    def keypoints_to_np(self, keypoints):
        return np.array([self.kp_to_np(kp) if kp is not None else np.array([-1,-1]) for kp in keypoints])

    def pose_results_to_keypoints_np(self, pose_results):
        return np.array([self.keypoints_to_np(pose_result.body.keypoints) for pose_result in pose_results])


if __name__ == "__main__":

    images1 = torch.load('/mckim/projects/MSU/facerec_framework_v4/assets/images.pt')
    images2 = torch.load('/mckim/projects/MSU/facerec_framework_v4/assets/images_ltcc.pt')
    images = torch.cat([images1, images2], dim=0)
    # images : tensor[24, 3, 384, 384] n=10616832 (40Mb) x∈[-1.000, 1.000] 
    unnorm_images_rgb = images * 0.5 + 0.5

    open_pose_det = OpenPoseDetectorBatched()

    # Timing infer_openpose_one_by_one
    start_time = time.time()
    N_repeat = 3
    # Timing infer_batch
    start_time = time.time()
    for _ in range(N_repeat):
        outputs2 = open_pose_det(unnorm_images_rgb)
    avg_time_batch = (time.time() - start_time) / N_repeat
    print(f"Average time for infer_batch: {avg_time_batch:.4f} seconds")
    draw_bodyposes(unnorm_images_rgb, outputs2, path='/mckim/temp/pose_batch.png')