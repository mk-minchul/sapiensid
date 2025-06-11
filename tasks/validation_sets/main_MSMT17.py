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
np.float = np.float_

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


def _process_dir(dir_path, list_path):
    with open(list_path, 'r') as txt:
        lines = txt.readlines()
    dataset = []
    pid_container = set()
    for img_idx, img_info in enumerate(lines):
        img_path, pid = img_info.split(' ')
        pid = int(pid) # no need to relabel
        camid = int(img_path.split('_')[2])
        img_path = osp.join(dir_path, img_path)
        clothes_id = -1
        dataset.append((img_path, pid, camid, clothes_id))
        pid_container.add(pid)
    num_imgs = len(dataset)
    num_pids = len(pid_container)
    # check if pid starts from 0 and increments with 1
    for idx, pid in enumerate(pid_container):
        assert idx == pid, "See code comment for explanation"
    return dataset, num_pids, num_imgs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/ssd2/data/body/personReID/MSMT17_V1')
    parser.add_argument('--dataset_name', type=str, default='msmt17_test')
    parser.add_argument('--save_dir', type=str, default='/ssd2/data/body/validation_sets')
    args = parser.parse_args()

    # Process the dataset
    test_dir = osp.join(args.dataset_path, 'test')
    list_query_path = osp.join(args.dataset_path, 'list_query.txt')
    list_gallery_path = osp.join(args.dataset_path, 'list_gallery.txt')
    query_dataset, num_query_pids, num_query_imgs = _process_dir(test_dir, list_query_path)
    gallery_dataset, num_gallery_pids, num_gallery_imgs = _process_dir(test_dir, list_gallery_path)
    num_pids = num_query_pids + num_gallery_pids

    # Print some statistics
    print(f"Number of identities: {num_pids}")


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
