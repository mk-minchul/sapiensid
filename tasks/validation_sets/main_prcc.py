import argparse
import glob
import os.path as osp
import random
from tqdm import tqdm
from PIL import Image
from datasets import Dataset
import torch
import os


def _process_dir_test(dataset_path):
    test_path = dataset_path
    pdirs = glob.glob(osp.join(test_path, '*'))
    pdirs.sort()

    pid_container = set()
    for pdir in glob.glob(osp.join(test_path, 'A', '*')):
        pid = int(osp.basename(pdir))
        pid_container.add(pid)
    pid_container = sorted(pid_container)
    pid2label = {pid:label for label, pid in enumerate(pid_container)}
    cam2label = {'A': 0, 'B': 1, 'C': 2}

    num_pids = len(pid_container)
    num_clothes = num_pids * 2

    query_dataset_same_clothes = []
    query_dataset_diff_clothes = []
    gallery_dataset = []
    for cam in ['A', 'B', 'C']:
        pdirs = glob.glob(osp.join(test_path, cam, '*'))
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                camid = cam2label[cam]
                if cam == 'A':
                    clothes_id = pid2label[pid] * 2
                    gallery_dataset.append((img_dir, pid, camid, clothes_id))
                elif cam == 'B':
                    clothes_id = pid2label[pid] * 2
                    query_dataset_same_clothes.append((img_dir, pid, camid, clothes_id))
                else:
                    clothes_id = pid2label[pid] * 2 + 1
                    query_dataset_diff_clothes.append((img_dir, pid, camid, clothes_id))

    pid2imgidx = {}
    for idx, (img_dir, pid, camid, clothes_id) in enumerate(gallery_dataset):
        if pid not in pid2imgidx:
            pid2imgidx[pid] = []
        pid2imgidx[pid].append(idx)

    # get 10 gallery index to perform single-shot test
    gallery_idx = {}
    random.seed(3)
    for idx in range(0, 10):
        gallery_idx[idx] = []
        for pid in pid2imgidx:
            gallery_idx[idx].append(random.choice(pid2imgidx[pid]))
             
    num_imgs_query_same = len(query_dataset_same_clothes)
    num_imgs_query_diff = len(query_dataset_diff_clothes)
    num_imgs_gallery = len(gallery_dataset)

    return query_dataset_same_clothes, query_dataset_diff_clothes, gallery_dataset, \
           num_pids, num_imgs_query_same, num_imgs_query_diff, num_imgs_gallery, \
           num_clothes, gallery_idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/ssd2/data/body/personReID/prcc/rgb/test')
    parser.add_argument('--dataset_name', type=str, default='prcc_test')
    parser.add_argument('--save_dir', type=str, default='/ssd2/data/body/validation_sets')
    args = parser.parse_args()

    # Process the dataset
    result = _process_dir_test(args.dataset_path)

    # Unpack the results
    query_same, query_diff, gallery, num_pids, num_query_same, num_query_diff, num_gallery, num_clothes, gallery_idx = result

    # Print some statistics
    print(f"Number of identities: {num_pids}")
    print(f"Number of query images (same clothes): {num_query_same}")
    print(f"Number of query images (different clothes): {num_query_diff}")
    print(f"Number of gallery images: {num_gallery}")
    print(f"Number of clothes: {num_clothes}")

    full_meta_data = query_same + query_diff + gallery
    full_meta_data_label = ['query_same'] * num_query_same + ['query_diff'] * num_query_diff + ['gallery'] * num_gallery
    meta_data = {
        'full_meta_data': full_meta_data,
        'full_meta_data_label': full_meta_data_label,
        'query_same': query_same,
        'query_diff': query_diff,
        'gallery': gallery,
        'num_pids': num_pids,
        'num_query_same': num_query_same,
        'num_query_diff': num_query_diff,
        'num_gallery': num_gallery,
        'num_clothes': num_clothes,
        'gallery_idx': gallery_idx
    }


    def entry_for_row(index, row):
        path, pid, camid, clothes_id = row
        image_pil = Image.open(path).convert('RGB')
        rel_path = osp.relpath(path, os.path.dirname(os.path.dirname(args.dataset_path)))
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
