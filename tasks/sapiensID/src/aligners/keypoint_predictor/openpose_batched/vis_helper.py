import numpy as np
from PIL import Image
import cv2
import math
import torch


def unnormalize(tensor):
    tensor = tensor * 0.5 + 0.5
    tensor = tensor * 255
    tensor = tensor.clamp(0, 255)
    tensor = tensor.type(torch.uint8)
    return tensor


def tensor_to_pil(tensor, tensor_is_normalized=True):
    if tensor_is_normalized:
        tensor = unnormalize(tensor)
    # to uint8
    tensor = tensor.type(torch.uint8)
    if tensor.ndim == 4:
        tensor = tensor.cpu().numpy()
        tensor = np.transpose(tensor, (0, 2, 3, 1))
        if tensor.shape[3] == 1:
            tensor = tensor[:, :, :, 0]
        pils = [Image.fromarray(t) for t in tensor]
    elif tensor.ndim == 3:
        tensor = tensor.cpu().numpy()
        tensor = np.transpose(tensor, (1, 2, 0))
        pil = Image.fromarray(tensor)
        pils = [pil]
    else:
        raise NotImplementedError
    return pils




def stack_images(images, num_cols, num_rows, pershape=(112,112), border=False):
    stack = []
    for rownum in range(num_rows):
        row = []
        for colnum in range(num_cols):
            idx = rownum * num_cols + colnum
            if idx > len(images)-1:
                img_resized = np.ones((pershape[0], pershape[1], 3)) * 255
            else:
                if isinstance(images[idx], str):
                    img = cv2.imread(images[idx])
                    img_resized = cv2.resize(img, dsize=pershape)
                elif isinstance(images[idx], Image.Image):
                    if border:
                        # put border
                        from PIL import ImageOps
                        img_resized = ImageOps.expand(images[idx], border=1, fill='black')
                        img_resized = img_resized.resize(pershape[::-1])
                    else:
                        img_resized = images[idx].resize(pershape[::-1])
                    img_resized = np.array(img_resized)
                else:
                    img_resized = cv2.resize(images[idx], dsize=pershape)
            row.append(img_resized)
        row = np.concatenate(row, axis=1)
        stack.append(row)
    stack = np.concatenate(stack, axis=0)
    return stack



def draw_one_bodypose(canvas, keypoints) -> np.ndarray:
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


def draw_bodyposes(unnorm_images_rgb, pose_results, path='/mckim/temp/pose.png'):
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
    pose_images_vis.save(path)
    x = (unnorm_images_rgb - 0.5) / 0.5
    orig_images_vis = stack_images(tensor_to_pil(x), 4,4, pershape=(512,512))
    orig_images_vis = Image.fromarray(orig_images_vis)
    orig_images_vis.save(path.replace('.png', '_orig.png'))