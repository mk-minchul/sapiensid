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

import argparse
from torch.utils.data import DataLoader
from general_utils.os_utils import get_all_files
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from dataset_helper import make_dataset
from general_utils.img_utils import tensor_to_pil, stack_images
from writer import Writer, ArrayWriter
import pandas as pd
import lovely_tensors as lt
from PIL import Image
from tqdm import tqdm
lt.monkey_patch()
import torch
from general_utils.img_utils import visualize, pil_to_tensor

def str2bool(v):
    return v.lower() in ('true', '1')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_record_root', type=str, default='/hdd3/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/raw_img_parquets_640_yolo_cropped')
    parser.add_argument('--cluster_result_path', type=str, default='/ssd2/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/regroup_result_v2/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320/cluster_eps:0.7_min_samples:2_max_group_per_label:-1/facefilter_0.95_merge_threshold_0.7_ambiguous_threshold_0.6_val_sim_threshold_0.7/cluster_final.csv')
    parser.add_argument('--save_root', type=str, default='/ssd2/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/')
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--vis_every', type=int, default=100)
    parser.add_argument('--mock', type=str2bool, default='false')
    args = parser.parse_args()

    cluster_df = pd.read_csv(args.cluster_result_path)

    # sort by remapped_label
    cluster_df = cluster_df.sort_values(by='remapped_label')
    using_paths = cluster_df['global_path'].tolist()
    using_labels = cluster_df['remapped_label'].tolist()
    unique_using_label = list(set(using_labels))
    label_mapping = dict(zip(unique_using_label, np.arange(len(unique_using_label))))

    del cluster_df
    path_to_label = {using_paths[i]: label_mapping[using_labels[i]] for i in range(len(using_paths))}

    # create save dir
    save_name = 'whole_body_bundle/' + os.path.basename(args.img_record_root) + \
        '/' + os.path.basename(os.path.dirname(os.path.dirname(args.cluster_result_path))) + \
        '/' + os.path.basename(os.path.dirname(args.cluster_result_path)) + \
        f'_image_size_{args.image_size}'
    save_dir = os.path.join(args.save_root, save_name)
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # create writer
    idx_header = ['local_idx', 'global_idx', 'detection_idx']
    kp_header = np.array([[f'kp_{i}_x' for i in range(17)], [f'kp_{i}_y' for i in range(17)]]).T.flatten().tolist()
    kp_writer = ArrayWriter(save_dir, 'kps_xyn.csv', idx_header, kp_header, n_digits=5)
    shape_writer = ArrayWriter(save_dir, 'shapes.csv', idx_header, ['height', 'width'], n_digits=0)
    writer = Writer(save_dir, prefix='train')

    img_record_paths = get_all_files(args.img_record_root, extension_list=['.rec'], sort=True)
    if args.mock :
        img_record_paths = img_record_paths[:2]

    n = 0
    for idx_img_path, img_rec_path in enumerate(img_record_paths):
        dataset = make_dataset([img_rec_path], -1, -1, path_to_label, image_size=args.image_size)
        def collate_fn(batch):
            try:
                # img, idx, path, crop_kp, label
                images = [x[0] for x in batch]
                indices = torch.tensor([x[1] for x in batch])
                paths = [x[2] for x in batch]
                crop_kps = torch.stack([x[3] for x in batch])
                labels = torch.tensor([x[4] for x in batch])
                orig_shapes = torch.tensor([x[5] for x in batch])
                return images, indices, paths, crop_kps, labels, orig_shapes
            except:
                return None, None, None, None, None, None
                

        if args.mock:
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn)
        else:
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, collate_fn=collate_fn)

        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'bundle {idx_img_path} / {len(img_record_paths)}', ncols=75):
            pils, indices, paths, crop_kps, labels, orig_shapes = batch
            if pils is None or paths[0] == -1:
                print('Error during batch making')
                continue
            for i, rgb_pil_img in enumerate(pils):
                kp = crop_kps[i]
                path = paths[i]
                label = labels[i].item()
                global_index, det_index = path.split('/')[-1].replace('.jpg', '').split('_')
                global_index, det_index = int(global_index), int(det_index)
                kp_writer.write_array(n, kp.cpu().numpy(), global_idx=global_index, detection_idx=det_index)
                shape_writer.write_array(n, orig_shapes[i].cpu().numpy(), global_idx=global_index, detection_idx=det_index)
                writer.write(rgb_pil_img=rgb_pil_img, save_path=path, label=label, bgr=False)
                writer.mark_done(n, path)
                n += 1
        
            if batch_idx % args.vis_every == 0 or args.mock:
                # find images whose label is equal to labes[0]
                most_frequent_label = np.argmax(np.bincount(labels.cpu().numpy()))
                same_label_indices = np.where(labels == most_frequent_label)[0]
                same_label_images = [pils[i] for i in same_label_indices]
                num_rows = min(max(1, len(same_label_images) // 8), 8)
                vis = stack_images(same_label_images, num_cols=8, num_rows=num_rows).astype(np.uint8)
                Image.fromarray(vis).save(f'{vis_dir}/{batch_idx}_{most_frequent_label}.png')
                try:
                    same_label_images_tensor = torch.stack([pil_to_tensor(img) for img in same_label_images])
                    same_label_kps = crop_kps[same_label_indices]
                    visualize(same_label_images_tensor, same_label_kps).save(f'{vis_dir}/{batch_idx}_{most_frequent_label}_with_ldmk.png')
                except:
                    print('Error during visualization')
                

    writer.close()
    kp_writer.close()
    shape_writer.close()

    with open(os.path.join(save_dir, 'done.txt'), 'w') as f:
        f.write('done')
    with open(os.path.join(save_dir, 'statistics.txt'), 'w') as f:
        info = pd.read_csv(os.path.join(save_dir, 'train.tsv'), sep='\t', header=None)
        info.columns = ['_idx', 'path', 'label']
        f.write(f'num_images: {len(info)}\n')
        f.write(f'num_labels: {len(info["label"].unique())}\n')
        f.write(f'num_images_per_label: {info["label"].value_counts().mean()}\n')
        f.write(f'smallest_label: {info["label"].min()}\n')
        f.write(f'largest_label: {info["label"].max()}\n')


if __name__ == '__main__':
    main()