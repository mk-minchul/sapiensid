from general_utils.img_utils import tensor_to_numpy, concat_pil
from PIL import Image
import cv2


def draw_ldmk_body(img, ldmk):
    if ldmk is None:
        return img
    colors = [
        (0, 255, 0),  # Original
        (85, 170, 0),  # Interpolated between (0, 255, 0) and (255, 0, 0)
        (170, 85, 0),  # Interpolated between (0, 255, 0) and (255, 0, 0)
        (255, 0, 0),  # Original
        (170, 0, 85),  # Interpolated between (255, 0, 0) and (0, 0, 255)
        (85, 0, 170),  # Interpolated between (255, 0, 0) and (0, 0, 255)
        (0, 0, 255),  # Original
        (85, 85, 170),  # Interpolated between (0, 0, 255) and (255, 255, 0)
        (170, 170, 85),  # Interpolated between (0, 0, 255) and (255, 255, 0)
        (255, 255, 0),  # Original
        (170, 255, 85),  # Interpolated between (255, 255, 0) and (0, 255, 255)
        (85, 255, 170),  # Interpolated between (255, 255, 0) and (0, 255, 255)
        (0, 255, 255),  # Original
        (85, 170, 255),  # Interpolated between (0, 255, 255) and (255, 0, 255)
        (170, 85, 255),  # Interpolated between (0, 255, 255) and (255, 0, 255)
        (255, 0, 255),  # Original
        (170, 0, 170)  # Interpolated between (255, 0, 255) and (0, 255, 0)
    ]
    img = img.copy()
    for i in range(len(ldmk)//2):
        color = colors[i]
        cv2.circle(img, (int(ldmk[i*2] * img.shape[1]),
                         int(ldmk[i*2+1] * img.shape[0])), 3, color, 4)
    return img

def visualize_body(tensor, ldmks=None, bboxes_xyxyn=None):
    assert tensor.ndim == 4
    images = [tensor_to_numpy(image_tensor) for image_tensor in tensor]
    if ldmks is not None:
        images = [draw_ldmk_body(images[j], ldmks[j].ravel()) for j in range(len(images))]
    if bboxes_xyxyn is not None:
        for j, bbox in enumerate(bboxes_xyxyn):
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            x, y, w, h = (int(x * images[j].shape[1]), int(y * images[j].shape[0]),
                          int(w * images[j].shape[1]), int(h * images[j].shape[0]))
            cv2.rectangle(images[j], (x, y), (x+w, y+h), (0, 255, 0), 2)
    pil_images = [Image.fromarray(im.astype('uint8')) for im in images]
    return concat_pil(pil_images)