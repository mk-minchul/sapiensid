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


def _process_dir(dir_path, relabel=False, label_start=0):
    img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')

    pid_container = set()
    for img_path in img_paths:
        pid, _ = map(int, pattern.search(img_path).groups())
        if pid == -1: continue  # junk images are just ignored
        pid_container.add(pid)
    pid2label = {pid:label for label, pid in enumerate(pid_container)}

    dataset = []
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1: continue  # junk images are just ignored
        if label_start == 0:
            assert 0 <= pid <= 1501  # pid == 0 means background
        assert 1 <= camid <= 6
        camid -= 1 # index starts from 0
        if relabel: pid = pid2label[pid] + label_start
        clothes_id = -1
        dataset.append((img_path, pid, camid, clothes_id))

    num_pids = len(pid_container)
    num_imgs = len(dataset)
    return dataset, num_pids, num_imgs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/ssd2/data/body/personReID/Market-1501-v15.09.15')
    parser.add_argument('--dataset_name', type=str, default='market_1501_test')
    parser.add_argument('--save_dir', type=str, default='/ssd2/data/body/validation_sets')
    args = parser.parse_args()

    # Process the dataset
    query_dir = osp.join(args.dataset_path, 'query')
    gallery_dir = osp.join(args.dataset_path, 'bounding_box_test')
    query_dataset, num_query_pids, num_query_imgs = _process_dir(query_dir, relabel=False)
    gallery_dataset, num_gallery_pids, num_gallery_imgs = _process_dir(gallery_dir, relabel=False)
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
