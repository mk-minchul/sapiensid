import os
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import numpy as np
np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_

from general_utils.os_utils import get_all_files
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import io
import torchvision
from general_utils.img_utils import tensor_to_pil, concat_pil
from torch.utils.data import Dataset, DataLoader
from writer import Writer, ArrayWriter
from tqdm import tqdm
import time
import datetime
import torch.distributed as distributed
from lightning.fabric import Fabric
from functools import partial
import argparse
import random
from torch.utils.data.distributed import DistributedSampler
from ultralytics import YOLO
import cv2

import lovely_tensors as lt
lt.monkey_patch()

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    # needed for speeding up dataloader in torch 2.0.1
    os.sched_setaffinity(0, range(os.cpu_count()))

def setup_dataloader_from_dataset(dataset,
                                  is_train,
                                  batch_size,
                                  num_workers,
                                  seed,
                                  fabric,
                                  collate_fn=None):

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=fabric.local_rank, seed=seed)

    if is_train:
        sampler = DistributedSampler(dataset=dataset, num_replicas=fabric.world_size,
                                     rank=fabric.local_rank, shuffle=True, drop_last=True, seed=seed)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
                                collate_fn=collate_fn, worker_init_fn=init_fn,
                                drop_last=True, shuffle=(sampler is None), pin_memory=True, )

    else:
        sampler = DistributedSampler(dataset=dataset, num_replicas=fabric.world_size,
                                     rank=fabric.local_rank, shuffle=False, drop_last=False, seed=seed)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
                                collate_fn=collate_fn, worker_init_fn=init_fn,
                                drop_last=False, shuffle=(sampler is None), pin_memory=False, )
    dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=False)
    return dataloader



class ParquetDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try:
            return self.read_row(self.dataframe.iloc[idx])
        except:
            return self.return_dummy()

    def read_row(self, row):
        img_data = row['jpg']
        label = row['label']
        index = row['idx']

        # Convert the jpg column to an image
        image = Image.open(io.BytesIO(img_data)).convert('RGB')
        image_tensor = self.transform(image)
        label_int = int(label)
        index_int = int(index)
        return image_tensor, label_int, index_int

    def return_dummy(self):
        print("Error reading row, returning dummy data")
        return torch.randn(3, 256, 256), 0, 0


def list_collate_fn(examples):
    image_list = [example[0] for example in examples]
    labels = [example[1] for example in examples]
    indices = [example[2] for example in examples]
    return image_list, labels, indices


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
        (170, 0, 170)  # Interpolated between (255, 0, 255) and (0, 255, 0)
    ]
    img = img.copy()
    for i in range(len(ldmk)//2):
        color = colors[i]
        cv2.circle(img, (int(ldmk[i*2] * img.shape[1]),
                         int(ldmk[i*2+1] * img.shape[0])), 3, color, 4)
    return img

def visualize(tensor, ldmks=None, bboxes_xyxyn=None):
    assert tensor.ndim == 4
    images = [tensor_to_numpy(image_tensor) for image_tensor in tensor]
    if ldmks is not None:
        images = [draw_ldmk(images[j], ldmks[j].ravel()) for j in range(len(images))]
    if bboxes_xyxyn is not None:
        for j, bbox in enumerate(bboxes_xyxyn):
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            x, y, w, h = (int(x * images[j].shape[1]), int(y * images[j].shape[0]),
                          int(w * images[j].shape[1]), int(h * images[j].shape[0]))
            cv2.rectangle(images[j], (x, y), (x+w, y+h), (0, 255, 0), 2)
    pil_images = [Image.fromarray(im.astype('uint8')) for im in images]
    return concat_pil(pil_images)

def tensor_to_numpy(tensor):
    # -1 to 1 tensor to 0-255
    arr = tensor.numpy().transpose(1,2,0)
    return (arr * 0.5 + 0.5) * 255

def pad_bbox(bbox, padding_ratio):
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    pad_x = padding_ratio * width
    pad_y = padding_ratio * height
    xmin, ymin, xmax, ymax = xmin-pad_x, ymin-pad_y, xmax+pad_x, ymax+pad_y
    return torch.tensor([xmin, ymin, xmax, ymax], device=bbox.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", type=str, default='/hdd2/data/faces/webface260m/WebFaceV2_debug/raw_img_parquets_640/1_parquet')
    parser.add_argument("--save_dir", type=str, default='/hdd2/data/faces/webface260m/WebFaceV2_debug/raw_img_parquets_640_yolo_cropped')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--confidence_threshold', type=float, default=0.7)
    start = time.time()
    args = parser.parse_args()
    # get all parquet files
    files = get_all_files(args.parquet_dir, extension_list=['.parquet'], sort=True)

    fabric = Fabric(precision='32-true',
                    loggers=[],
                    accelerator="auto",
                    strategy="ddp",
                    devices=args.num_gpu)
    fabric.seed_everything(args.seed)
    world_size = fabric.world_size
    local_rank = fabric.local_rank
    if world_size == 1:
        try:
            fabric.launch()
        except:
            pass
    fabric.setup_dataloader_from_dataset = partial(setup_dataloader_from_dataset, fabric=fabric, seed=args.seed)

    save_dir = os.path.join(args.save_dir, os.path.basename(args.parquet_dir)+f'_rank_{local_rank}')
    idx_header = ['local_idx', 'global_idx', 'detection_idx']
    body_kp_header = np.array([[f'kp_{i}_x' for i in range(17)], [f'kp_{i}_y' for i in range(17)]]).T.flatten().tolist()
    bbox_header = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
    confidence_header = ['confidence']

    body_kp_writer = ArrayWriter(save_dir, 'body_kps_xyn.csv', idx_header, body_kp_header, n_digits=5)
    bbox_writer = ArrayWriter(save_dir, 'bbox_xyxyn.csv', idx_header, bbox_header, n_digits=5)
    confidence_writer = ArrayWriter(save_dir, 'confidence.csv', idx_header, confidence_header, n_digits=5)
    writer = Writer(save_dir, prefix='train')
    vis_dir = os.path.join(save_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    model = YOLO("yolov8n-pose.pt")
    model.to(fabric.device)

    n = 0
    for file_idx, file in enumerate(files):
        print("Loading", file)
        df = pd.read_parquet(file)
        filtered_df = df.dropna(subset=['jpg'])
        filtered_df.index = range(len(filtered_df))

        dataset = ParquetDataset(filtered_df)
        dataloader = fabric.setup_dataloader_from_dataset(dataset=dataset,
                                                          is_train=False,
                                                          batch_size=128,
                                                          num_workers=0 if 'debug' in args.parquet_dir else 4,
                                                          collate_fn=None,
                                                          # collate_fn=list_collate_fn)
                                                          )

        batch = next(iter(dataloader))
        images, labels, indices = batch
        print('Data loaded')

        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Processing {file_idx}/{len(files)}',
                               disable=local_rank != 0):
            images, labels, indices = batch
            unnorm_images_rgb = (images.to(fabric.device) * 0.5 + 0.5)
            results = model(unnorm_images_rgb, device=fabric.device, verbose=False)

            # write the results
            for i, r in enumerate(results):
                keypoints = r.keypoints.xyn
                bounding_boxes_xyxyn = r.boxes.xyxyn
                confidences = r.boxes.conf
                rgb_img = Image.fromarray(r.orig_img)
                label = labels[i].item()

                for det_idx, (kp, bbox, conf) in enumerate(zip(keypoints, bounding_boxes_xyxyn, confidences)):
                    if conf < args.confidence_threshold:
                        continue
                    if det_idx > 5:
                        break
                    padded_bbox = pad_bbox(bbox, padding_ratio=0.05)
                    padded_bbox_abs = [int(padded_bbox[0] * rgb_img.width), int(padded_bbox[1] * rgb_img.height),
                                        int(padded_bbox[2] * rgb_img.width), int(padded_bbox[3] * rgb_img.height)]
                    body_kp_writer.write_array(n, kp.cpu().numpy(), global_idx=indices[i], detection_idx=det_idx)
                    bbox_writer.write_array(n, padded_bbox.cpu().numpy(), global_idx=indices[i], detection_idx=det_idx)
                    confidence_writer.write_array(n, np.array([conf.item()]), global_idx=indices[i], detection_idx=det_idx)
                    path = f'{label}/{indices[i]}_{det_idx}.jpg'
                    cropped = rgb_img.crop(padded_bbox_abs)
                    writer.write(rgb_pil_img=cropped, save_path=path, label=label, bgr=False)
                    writer.mark_done(n, path)

                    if n % 10000 == 0:
                        vis_path = os.path.join(vis_dir, f'vis_{n}.png')
                        visualize(images[i].unsqueeze(0).cpu(), kp.unsqueeze(0), padded_bbox.unsqueeze(0)).save(vis_path)
                        cropped.save(os.path.join(vis_dir, f'cropped_{n}.png'))
                        # Image.fromarray(r.plot()).save(os.path.join(vis_dir, f'full_{n}.png'))
                    n += 1


    # save a done.txt file
    with open(os.path.join(save_dir, 'done.txt'), 'w') as f:
        f.write('done\n')

    # close files
    body_kp_writer.close()
    bbox_writer.close()
    writer.close()
