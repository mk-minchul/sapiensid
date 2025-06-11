import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys
sys.path.append(os.path.join(root))
import numpy as np
np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_

from datasets import Dataset
import torch
from functools import partial
import argparse
from general_utils.config_utils import load_config
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import lovely_tensors as lt
lt.monkey_patch()
from general_utils.img_utils import visualize
from array_writer import ArrayPathWriter

def preprocess_transform(examples, image_transforms, keypoints_df=None):
    images = [image.convert("RGB") for image in examples['image']]
    images = [image_transforms(image) for image in images]
    examples["pixel_values"] = images

    shapes = [(image.size[1], image.size[0]) for image in examples['image']] # (height, width)
    examples['shapes'] = shapes

    if keypoints_df is not None:
        keypoints = keypoints_df.loc[examples['path']].to_numpy()
        examples['keypoints'] = keypoints

    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    indexes = torch.tensor([example["index"] for example in examples], dtype=torch.int)
    image_paths = [example["path"] for example in examples]

    result = {
        "pixel_values": pixel_values,
        "index": indexes,
        "image_paths": image_paths,
    }

    shapes = torch.tensor([example["shapes"] for example in examples], dtype=torch.int)
    result['shapes'] = shapes
    if 'keypoints' in examples[0]:
        keypoints = torch.tensor([example["keypoints"] for example in examples], dtype=torch.float)
        result['keypoints'] = keypoints
    
    return result


class SquarePad:
    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        c, h, w = img.shape
        max_wh = max(w, h)
        hp = max_wh - h
        wp = max_wh - w
        padding = (wp // 2, hp // 2, wp - (wp // 2), hp - (hp // 2))
        return F.pad(img, padding, self.fill, self.padding_mode)



def main(args):

    np.random.seed(2222)
    torch.manual_seed(2222)
    torch.cuda.manual_seed_all(2222)
    print(f"Seed set to {2222}")

    aligner_config_name = args.aligner_config_name
    task_name = args.task_name
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    num_workers = args.num_workers

    rgb_mean = [0.5, 0.5, 0.5]
    rgb_std = [0.5, 0.5, 0.5]
    input_size = 384
    device = 'cuda'

    aligner_config = load_config(os.path.join(root, 'tasks', task_name, f'src/aligners/configs/{aligner_config_name}'))
    aligner_config.rgb_mean = rgb_mean
    aligner_config.rgb_std = rgb_std
    aligner_module = __import__(f'tasks.{task_name}.aligners', fromlist=['get_aligner'])
    get_aligner = getattr(aligner_module, 'get_aligner')
    aligner = get_aligner(aligner_config)
    aligner.eval()
    # aligner.face_predictor.eval()
    # aligner.faceness_threshold = 0.0
    aligner.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        SquarePad(fill=1),
        transforms.Resize(input_size, antialias=None),
        transforms.Normalize(mean=rgb_mean, std=rgb_std),
    ])

    dataset = Dataset.load_from_disk(dataset_path)
    preprocess = partial(preprocess_transform, image_transforms=transform)
    dataset = dataset.with_transform(preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    kp_header = np.array([[f'kp_{i}_x' for i in range(22)], [f'kp_{i}_y' for i in range(22)]]).T.flatten().tolist()
    kp_writer = ArrayPathWriter(save_root=dataset_path,
                             filename='kps_22_xyn.csv',
                             idx_header=['local_idx', 'path'],
                             array_header=kp_header,
                             n_digits=5)
    shape_writer = ArrayPathWriter(save_root=dataset_path,
                             filename='shapes.csv',
                             idx_header=['local_idx', 'path'],
                             array_header=['height', 'width'],
                             n_digits=5)
    os.makedirs(os.path.join(dataset_path, 'kp_vis'), exist_ok=True)

    n = 0
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Extracting Feature',):
        images = batch['pixel_values']
        paths = batch['image_paths']
        keypoints, foreground_masks = aligner(images.to(device))
        shapes = batch['shapes']

        if batch_idx % 1000 == 0:
            visualize(images, keypoints, ncols=8, nrows=4).save(os.path.join(dataset_path, f'kp_vis/{batch_idx}.png'))

        for j in range(len(images)):
            path = paths[j]
            kps = keypoints[j].cpu()
            shape = shapes[j].cpu()
            kp_writer.write_array(n, path, kps)
            shape_writer.write_array(n, path, shape)
            n += 1

    kp_writer.close()
    shape_writer.close()
    with open(os.path.join(dataset_path, 'complete_kp_22.txt'), 'w') as f:
        f.write('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aligner_config_name', type=str, default='openpose_dfa.yaml')
    parser.add_argument('--task_name', type=str, default='sapiensID')
    parser.add_argument('--dataset_path', type=str, default='/ssd2/data/body/validation_sets/ltcc_test')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    main(args)
