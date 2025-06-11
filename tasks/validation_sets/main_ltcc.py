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



def _process_dir_test(query_path, gallery_path):
    query_img_paths = glob.glob(osp.join(query_path, '*.png'))
    gallery_img_paths = glob.glob(osp.join(gallery_path, '*.png'))
    query_img_paths.sort()
    gallery_img_paths.sort()
    pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
    pattern2 = re.compile(r'(\w+)_c')

    pid_container = set()
    clothes_container = set()
    for img_path in query_img_paths:
        pid, _, _ = map(int, pattern1.search(img_path).groups())
        clothes_id = pattern2.search(img_path).group(1)
        pid_container.add(pid)
        clothes_container.add(clothes_id)
    for img_path in gallery_img_paths:
        pid, _, _ = map(int, pattern1.search(img_path).groups())
        clothes_id = pattern2.search(img_path).group(1)
        pid_container.add(pid)
        clothes_container.add(clothes_id)
    pid_container = sorted(pid_container)
    clothes_container = sorted(clothes_container)
    pid2label = {pid:label for label, pid in enumerate(pid_container)}
    clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

    num_pids = len(pid_container)
    num_clothes = len(clothes_container)

    query_dataset = []
    gallery_dataset = []
    for img_path in query_img_paths:
        pid, _, camid = map(int, pattern1.search(img_path).groups())
        clothes_id = pattern2.search(img_path).group(1)
        camid -= 1 # index starts from 0
        clothes_id = clothes2label[clothes_id]
        query_dataset.append((img_path, pid, camid, clothes_id))

    for img_path in gallery_img_paths:
        pid, _, camid = map(int, pattern1.search(img_path).groups())
        clothes_id = pattern2.search(img_path).group(1)
        camid -= 1 # index starts from 0
        clothes_id = clothes2label[clothes_id]
        gallery_dataset.append((img_path, pid, camid, clothes_id))
    
    num_imgs_query = len(query_dataset)
    num_imgs_gallery = len(gallery_dataset)

    return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/ssd2/data/body/personReID/LTCC_ReID')
    parser.add_argument('--dataset_name', type=str, default='ltcc_test')
    parser.add_argument('--save_dir', type=str, default='/ssd2/data/body/validation_sets')
    args = parser.parse_args()

    # Process the dataset
    query_path = osp.join(args.dataset_path, 'query')
    gallery_path = osp.join(args.dataset_path, 'test')
    result = _process_dir_test(query_path, gallery_path)

    # Unpack the results
    query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes = result

    # Print some statistics
    print(f"Number of identities: {num_pids}")
    print(f"Number of clothes: {num_clothes}")


    full_meta_data = query_dataset + gallery_dataset
    full_meta_data_label = ['query'] * num_imgs_query + ['gallery'] * num_imgs_gallery
    meta_data = {
        'full_meta_data': full_meta_data,
        'full_meta_data_label': full_meta_data_label,
        'query': query_dataset,
        'gallery': gallery_dataset,
        'num_pids': num_pids,
        'num_query': num_imgs_query,
        'num_gallery': num_imgs_gallery,
        'num_clothes': num_clothes,
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
