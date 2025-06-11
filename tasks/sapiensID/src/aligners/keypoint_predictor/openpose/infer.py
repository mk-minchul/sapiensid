from controlnet_aux.util import HWC3, resize_image
from controlnet_aux.open_pose import draw_poses
from general_utils.img_utils import stack_images, visualize, tensor_to_pil
from controlnet_aux.processor import Processor
from controlnet_aux import OpenposeDetector
from PIL import Image
import numpy as np
import cv2
from controlnet_aux.open_pose import PoseResult, BodyResult

def infer_openpose_one_by_one(unnorm_images_rgb, open_pose_detector):
    assert isinstance(open_pose_detector, OpenposeDetector)

    pose_results = []
    for i in range(unnorm_images_rgb.shape[0]):
        input_image = unnorm_images_rgb[i].permute(1,2,0).cpu() * 255
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, 512)
        H, W, C = input_image.shape
        poses = open_pose_detector.detect_poses(input_image, include_hand=False, include_face=False)
        # order according to body_score
        if len(poses) > 0:
            pose_result = sorted(poses, key=lambda x: x.body.total_score, reverse=True)[0]  # descending
        else:
            # make null pose
            pose_result = PoseResult(body=BodyResult(keypoints=[None]*18, total_score=0, total_parts=0), face=None, left_hand=None, right_hand=None)
        pose_results.append(pose_result)
    return pose_results


def draw_bodyposes(unnorm_images_rgb, pose_results):
    pose_images = []
    B, C, H, W = unnorm_images_rgb.shape

    for i in range(B):
        pose_result = pose_results[i]
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        canvas = draw_one_bodypose(canvas, pose_result.body.keypoints)
        # canvas = draw_one_facepose(canvas, pose_result.face)
        canvas = Image.fromarray(canvas)
        pose_images.append(canvas)

    pose_images_vis = stack_images(pose_images, 4, 4, pershape=(512,512))
    pose_images_vis = Image.fromarray(pose_images_vis)
    pose_images_vis.save('/mckim/temp/pose.png')
    x = (unnorm_images_rgb - 0.5) / 0.5
    orig_images_vis = stack_images(tensor_to_pil(x), 4,4, pershape=(512,512))
    orig_images_vis = Image.fromarray(orig_images_vis)
    orig_images_vis.save('/mckim/temp/orig.png')

def draw_with_keypoints_4x4(keypoints, x):
    from general_utils.img_utils import visualize
    v1 = visualize(x.cpu()[:4], keypoints.cpu()[:4])
    v2 = visualize(x.cpu()[4:8], keypoints.cpu()[4:8])
    v3 = visualize(x.cpu()[8:12], keypoints.cpu()[8:12])
    v4 = visualize(x.cpu()[12:16], keypoints.cpu()[12:16])
    v = np.concatenate([v1,v2,v3,v4], axis=0)
    Image.fromarray(v).save('/mckim/temp/keypoints_yolo.png')


def kp_to_np(kp):
    return np.array([kp.x, kp.y])

def keypoints_to_np(keypoints):
    return np.array([kp_to_np(kp) if kp is not None else np.array([-1,-1]) for kp in keypoints])

def pose_results_to_keypoints_np(pose_results):
    return np.array([keypoints_to_np(pose_result.body.keypoints) for pose_result in pose_results])


def draw_one_bodypose(canvas, keypoints) -> np.ndarray:
    import cv2
    import math
    H, W, C = canvas.shape
    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for idx_k, (keypoint, color) in enumerate(zip(keypoints, colors)):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)
        # add the keypoint number
        cv2.putText(canvas, str(idx_k), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return canvas


def draw_one_facepose(canvas, keypoints) -> np.ndarray:
    eps = 0.01
    """
    Draw keypoints representing face pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the face pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the face keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn face pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """    
    if not keypoints:
        return canvas
    
    H, W, C = canvas.shape
    for keypoint in keypoints:
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas