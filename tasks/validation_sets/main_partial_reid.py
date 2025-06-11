import argparse
import glob
import os.path as osp
import random
from tqdm import tqdm
from PIL import Image
from datasets import Dataset
import torch
import os
import re

def process_dir(dir_path, relabel=False, is_query=True):
    img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    if is_query:
        camid = 0
    else:
        camid = 1
    pid_container = set()
    for img_path in img_paths:
        img_name = img_path.split('/')[-1]
        pid = int(img_name.split('_')[0])
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    data = []
    for img_path in img_paths:
        img_name = img_path.split('/')[-1]
        pid = int(img_name.split('_')[0])
        if relabel:
            pid = pid2label[pid]
        clothes_id = -1
        data.append((img_path, pid, camid, clothes_id))
    return data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/ssd2/data/body/personReID/Partial-REID_Dataset')
    parser.add_argument('--dataset_name', type=str, default='partial_reid_test')
    parser.add_argument('--save_dir', type=str, default='/ssd2/data/body/validation_sets')
    args = parser.parse_args()

    # Process the dataset
    query_dir = osp.join(args.dataset_path, 'occluded_body_images')
    gallery_dir = osp.join(args.dataset_path, 'whole_body_images')

    query_dataset = process_dir(query_dir, relabel=False)
    gallery_dataset = process_dir(gallery_dir, relabel=False, is_query=False)

    # Print some statistics
    num_query_imgs = len(query_dataset)
    num_gallery_imgs = len(gallery_dataset)
    num_pids = len(set([pid for _, pid, _, _ in query_dataset]))
    print(f"Number of identities: {num_pids}")
    print(f"Number of query images: {num_query_imgs}")
    print(f"Number of gallery images: {num_gallery_imgs}")

    full_meta_data = query_dataset + gallery_dataset
    full_meta_data_label = ['query'] * num_query_imgs + ['gallery'] * num_gallery_imgs
    meta_data = {
        'full_meta_data': full_meta_data,
        'full_meta_data_label': full_meta_data_label,
        'query': query_dataset,
        'gallery': gallery_dataset,
        'num_pids': num_pids,
        'num_query': num_query_imgs,
        'num_gallery': num_gallery_imgs,
    }


    def entry_for_row(index, row):
        path, pid, camid, clothes_id = row
        image_pil = Image.open(path).convert('RGB')
        rel_path = osp.relpath(path, args.dataset_path)
        return {
            "image": image_pil,
            "index": index,
            "path": rel_path,
            "pid": pid,
            "camid": camid,
            "clothes_id": clothes_id
        }

    def generator():
        for index, row in tqdm(enumerate(full_meta_data), total=len(full_meta_data)):
            yield entry_for_row(index, row)

    dataset_name = args.dataset_name
    ds = Dataset.from_generator(generator)
    os.makedirs(os.path.join(args.save_dir, dataset_name), exist_ok=True)
    ds.save_to_disk(os.path.join(args.save_dir, dataset_name), num_shards=1)
    torch.save(meta_data, os.path.join(args.save_dir, dataset_name, 'metadata.pt'))
