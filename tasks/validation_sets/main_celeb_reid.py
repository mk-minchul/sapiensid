import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys

import numpy as np
import os.path as osp
from scipy.io import loadmat
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


class CelebReID(object):
    """ CelebReID

    Reference:
        Xu et al. CelebReID: A Long-Term Person Re-Identification Benchmark. arXiv:2105.14685, 2021.

    URL: https://github.com/PengBoXiangShang/CelebReID
    """
    dataset_dir = 'Celeb-reID'
    def __init__(self, root='data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_query_dir = osp.join(self.dataset_dir, 'query')
        self.test_gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self._check_before_run()
        train, num_train_pids, num_train_clothes = self._process_dir(self.train_dir)
        test_query, num_test_query_pids, num_test_query_clothes = self._process_dir(self.test_query_dir)
        test_gallery, num_test_gallery_pids, num_test_gallery_clothes = self._process_dir(self.test_gallery_dir)

        print("=> CelebReID loaded")
        print("Dataset statistics:")
        print("  --------------------------------------------")
        print("  subset        | # ids | # images | # clothes")
        print("  ----------------------------------------")
        print("  train         | {:5d} | {:8d} | {:9d} ".format(num_train_pids, len(train), num_train_clothes))
        print("  query         | {:5d} | {:8d} | {:9d} ".format(num_test_query_pids, len(test_query), num_test_query_clothes))
        print("  gallery       | {:5d} | {:8d} | {:9d} ".format(num_test_gallery_pids, len(test_gallery), num_test_gallery_clothes))
        print("  --------------------------------------------")

        self.train = train
        self.query = test_query
        self.gallery = test_gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.num_test_pids = num_test_query_pids

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def get_pid2label_and_clothes2label(self, img_names1, img_names2=None):
        if img_names2 is not None:
            img_names = img_names1 + img_names2
        else:
            img_names = img_names1

        pid_container = set()
        clothes_container = set()
        for img_name in img_names:
            names = img_name.split('.')[0].split('_')
            clothes = names[0] + names[2]
            pid = int(names[0][1:])
            pid_container.add(pid)
            clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

        if img_names2 is not None:
            return pid2label, clothes2label

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_name in img_names:
            names = img_name.split('.')[0].split('_')
            clothes = names[0] + names[2]
            pid = int(names[0][1:])
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            pid2clothes[pid, clothes_id] = 1

        return pid2label, clothes2label, pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_query_dir):
            raise RuntimeError("'{}' is not available".format(self.test_query_dir))
        if not osp.exists(self.test_gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.test_gallery_dir))



    def _process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        unique_pids = set()
        dataset = []
        for img_path in img_paths:
            pid, _, _ = img_path.split('_')
            unique_pids.add(pid)
            camid = -1
            clothes_id = -1
            dataset.append((img_path, 'celeb_'+str(pid), camid, clothes_id))

        num_pids = len(unique_pids)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/ssd2/data/body/personReID/')
    parser.add_argument('--dataset_name', type=str, default='CelebReID_test')
    parser.add_argument('--save_dir', type=str, default='/ssd2/data/body/validation_sets')
    args = parser.parse_args()

    dataset = CelebReID(root=args.dataset_path)

    # Print some statistics
    print(f"Number of identities: {dataset.num_test_pids}")
    print(f"Number of query images: {len(dataset.query)}")
    print(f"Number of gallery images: {len(dataset.gallery)}")

    full_meta_data = dataset.query + dataset.gallery
    full_meta_data_label = ['query'] * len(dataset.query) + ['gallery'] * len(dataset.gallery)
    meta_data = {
        'full_meta_data': full_meta_data,
        'full_meta_data_label': full_meta_data_label,
        'query': dataset.query,
        'gallery': dataset.gallery,
        'num_pids': dataset.num_test_pids,
        'num_query': len(dataset.query),
        'num_gallery': len(dataset.gallery),
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
