import cv2
import numpy as np
import torch
from PIL import Image
import io
import matplotlib.pyplot as plt
import math

def read_image_BGR(path):
    img = cv2.imread(path)
    # make 3 channel image
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]
    return img


def read_image_RGB(path):
    pil_img = Image.open(path).convert('RGB')
    img = np.array(pil_img)
    return img


def read_image_gray(path):
    pil_img = Image.open(path).convert('L')
    img = np.array(pil_img)
    return img


def save_image_BGR(img, path='./temp.jpg'):
    cv2.imwrite(path, img)


def save_image_RGB(img, path='./temp.jpg'):
    _img = Image.fromarray(img.astype(np.uint8))
    _img.save(path)


def save_image_gray(img, path='./temp.jpg'):
    _img = Image.fromarray(img)
    _img.save(path)


def resize_as(src_img, dst_img, interpolation=cv2.INTER_NEAREST):
    src_img = cv2.resize(src_img, dst_img.shape[:2][::-1], interpolation=interpolation)
    return src_img


def resize(src_img, size, interpolation=cv2.INTER_NEAREST):
    src_img = cv2.resize(src_img, size, interpolation=interpolation)
    return src_img


def gray_to_rgb(image):
    return np.stack([image] * 3, axis=-1)


def put_text(image, text, font_scale=2, font_thickness=3, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = 10
    text_y = 10 + text_size[1]
    texted_image = cv2.putText(image.copy(), text, (text_x, text_y), font, font_scale, color, font_thickness)
    return texted_image


def to_numpy(tensor, normalize=True):
    img = tensor.permute(1, 2, 0).numpy()
    if normalize:
        img = (img * 0.5 + 0.5) * 255
    return img.astype(np.uint8)

def numpy_to_tensor(img, normalize=True):
    assert img.ndim == 3
    assert img.shape[2] == 3
    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = (img / 255)
    elif img.dtype == np.float32:
        pass
    else:
        img = img.astype(np.float32)

    if normalize:
        img = (img - 0.5) / 0.5
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

def to_numpy_batched(tensor, normalize=True):
    img = tensor.permute(0, 2, 3, 1).numpy()
    if normalize:
        img = (img * 0.5 + 0.5) * 255
    return img.astype(np.uint8)


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




def prepare_text_img(text, height=300, width=30, fontsize=16, textcolor='C1', fontweight='normal', bg_color='white'):
    text_kwargs = dict(ha='center', va='center', fontsize=fontsize, color=textcolor, fontweight=fontweight)
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(width*px, height*px), facecolor=bg_color)
    plt.text(0.5, 0.5, text, **text_kwargs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_facecolor(bg_color)
    array = get_img_from_fig(fig)
    plt.clf()
    plt.cla()
    return array



def get_img_from_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def text_linebreak(text, by=4):
    text_split = text.split(' ')
    text_chunk = [" ".join(text_split[i:i + by]) for i in range(0, len(text_split), by)]
    text_with_linebreak = '\n'.join(text_chunk)
    return text_with_linebreak


def concat_pil(list_of_pil, axis=1):
    w, h = list_of_pil[0].size
    if axis == 0:
        new_im = Image.new('RGB', (w, h * len(list_of_pil)))
        for i, im in enumerate(list_of_pil):
            new_im.paste(im, (0, i * h))
    elif axis == 1:
        new_im = Image.new('RGB', (w * len(list_of_pil), h))
        for i, im in enumerate(list_of_pil):
            new_im.paste(im, (i * w, 0))
    else:
        raise NotImplementedError
    return new_im


def put_text_pil(pil_image, text, font_scale=2, font_thickness=3, color=(255, 255, 255)):
    array = np.array(pil_image)[..., ::-1]
    array = put_text(array, text, font_scale=font_scale, font_thickness=font_thickness, color=color)
    pil_image = Image.fromarray(array[..., ::-1])
    return pil_image


def visualize_tensor(tensor, ncols=8, nrows=4, path='./temp.png', flip_channel=True):
    assert tensor.ndim == 4
    images = [tensor_to_numpy(image_tensor) for image_tensor in tensor]
    vis = stack_images(images, num_cols=ncols, num_rows=nrows)
    if flip_channel:
        cv2.imwrite(path, vis[:,:,::-1])
    else:
        cv2.imwrite(path, vis)


def visualize_tensor_with_landmark(tensor, ldmks, ncols=8, nrows=4, path='./temp.png', flip_channel=True):
    assert tensor.ndim == 4
    images = [tensor_to_numpy(image_tensor) for image_tensor in tensor]
    from references.single_face_aligner.test_aligner import aligner_helper
    images = [aligner_helper.draw_ldmk(images[j], ldmks[j].ravel()) for j in range(len(images))]
    vis = stack_images(images, num_cols=ncols, num_rows=nrows)
    if flip_channel:
        cv2.imwrite(path, vis[:,:,::-1])
    else:
        cv2.imwrite(path, vis)


def tensor_to_numpy(tensor):
    # -1 to 1 tensor to 0-255
    arr = tensor.numpy().transpose(1,2,0)
    return (arr * 0.5 + 0.5) * 255


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
                    if border:
                        # Add border to numpy array
                        img_resized = cv2.copyMakeBorder(
                            images[idx],
                            top=1, bottom=1, left=1, right=1,
                            borderType=cv2.BORDER_CONSTANT,
                            value=[0, 0, 0]  # black border
                        )
                        # Resize back to desired shape to maintain consistent size
                        img_resized = cv2.resize(img_resized, dsize=pershape)
                    else:
                        img_resized = cv2.resize(images[idx], dsize=pershape)
            row.append(img_resized)
        row = np.concatenate(row, axis=1)
        stack.append(row)
    stack = np.concatenate(stack, axis=0)
    return stack

def draw_ldmk(img, ldmk):
    import cv2
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
        (170, 0, 170),  # Interpolated between (255, 0, 255) and (0, 255, 0)
        (255, 85, 170),  # New: Interpolated between (255, 0, 255) and (255, 85, 0)
        (85, 255, 85),  # New: Interpolated between (0, 255, 0) and (0, 85, 255)
        (255, 170, 85),  # New: Interpolated between (255, 85, 0) and (85, 170, 255)
        (0, 85, 255),   # New: Interpolated between (0, 255, 255) and (0, 85, 255)
        (170, 255, 255),  # New: Interpolated between (170, 85, 255) and (255, 255, 0)
        (255, 85, 85),   # New: Interpolated between (255, 85, 255) and (255, 85, 0)
        (85, 170, 0),  # Interpolated between (0, 255, 0) and (255, 0, 0)
        (170, 85, 0),  # Interpolated between (0, 255, 0) and (255, 0, 0)
        (255, 0, 0),  # Original
        (170, 0, 85),  # Interpolated between (255, 0, 0) and (0, 0, 255)
        (85, 0, 170),  # Interpolated between (255, 0, 0) and (0, 0, 255)
    ] * 10
    img = img.copy()
    for i in range(len(ldmk)//2):
        color = colors[i]
        if len(ldmk) == 44:
            right_indices = np.array([2, 4, 6, 8, 10, 12, 14, 16, 17, 20])
        elif len(ldmk) == 42:
            right_indices = np.array([2, 4, 6, 8, 10, 12, 14, 16, 17, 19])
        elif len(ldmk) == 34:
            right_indices = np.array([2, 4, 6, 8, 10, 12, 14, 16])
        else: 
            right_indices = []
        if i in right_indices:
            # draw square
            top_left = (int(ldmk[i*2] * img.shape[1]) - 4, int(ldmk[i*2+1] * img.shape[0]) - 4)
            bottom_right = (int(ldmk[i*2] * img.shape[1]) + 4, int(ldmk[i*2+1] * img.shape[0]) + 4)
            cv2.rectangle(img, top_left, bottom_right, color, 2, lineType=cv2.LINE_8)
        else:
            cv2.circle(img, (int(ldmk[i*2] * img.shape[1]),
                            int(ldmk[i*2+1] * img.shape[0])), 3, color, 4)
    return img


def visualize(tensor, ldmks=None, axis=1, bboxes_xyxyn=None, texts=None, ncols=None, nrows=None, pershape=(160,160), border=False):
    assert tensor.ndim == 4
    images = [tensor_to_numpy(image_tensor.detach().cpu().float()) for image_tensor in tensor]
    if ldmks is not None:
        images = [draw_ldmk(images[j], ldmks[j].ravel()) for j in range(len(images))]
    if bboxes_xyxyn is not None:
        for j, bbox in enumerate(bboxes_xyxyn):
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            x, y, w, h = (int(x * images[j].shape[1]), int(y * images[j].shape[0]),
                          int(w * images[j].shape[1]), int(h * images[j].shape[0]))
            images[j] = images[j].astype(np.uint8).copy()
            cv2.rectangle(images[j], (x, y), (x+w, y+h), (0, 255, 0), 2)
    if texts is not None:
        for j, text in enumerate(texts):
            images[j] = put_text(images[j], text, font_scale=2, font_thickness=2, color=(255, 255))
    pil_images = [Image.fromarray(im.astype('uint8')) for im in images]
    if ncols is not None or nrows is not None:
        return Image.fromarray(stack_images(images, ncols, nrows, pershape=pershape, border=border).astype(np.uint8))
    else:
        return concat_pil(pil_images, axis)

def pil_to_tensor(pil_img):
    x = torch.from_numpy(np.array(pil_img)[:, :, :3]).permute(2, 0, 1).float() / 255.0
    return ((x - 0.5) / 0.5).clip(-1, 1)



def tile_tensor(tensor):
    """
    Tiles a tensor of shape (B, 3, H, W) into a grid that is as square as possible.
    Any missing grid cells are filled with ones.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, 3, H, W).

    Returns:
        torch.Tensor: Tiled tensor of shape (3, grid_rows * H, grid_cols * W).
    """
    B, C, H, W = tensor.shape

    # Determine grid dimensions: use floor(sqrt(B)) rows and ceil(B / rows) columns.
    grid_rows = int(math.floor(math.sqrt(B)))
    grid_cols = int(math.ceil(B / grid_rows))

    # Calculate how many cells are needed in total and how many are missing.
    grid_size = grid_rows * grid_cols
    missing = grid_size - B

    # If there are missing cells, create a filler tensor filled with ones.
    if missing > 0:
        filler = torch.ones((missing, C, H, W), dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, filler], dim=0)

    # Reshape the tensor to (grid_rows, grid_cols, C, H, W)
    tensor = tensor.view(grid_rows, grid_cols, C, H, W)
    
    # Permute to bring the channels in front and combine grid rows with H, grid cols with W.
    tensor = tensor.permute(2, 0, 3, 1, 4).contiguous()
    tiled = tensor.view(C, grid_rows * H, grid_cols * W)
    
    return tiled
