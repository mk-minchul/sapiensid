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


class DeepChange(object):
    """ DeepChange

    Reference:
        Xu et al. DeepChange: A Long-Term Person Re-Identification Benchmark. arXiv:2105.14685, 2021.

    URL: https://github.com/PengBoXiangShang/deepchange
    """
    dataset_dir = 'DeepChange'
    def __init__(self, root='data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train-set')
        self.train_list = osp.join(self.dataset_dir, 'train-set-bbox.txt')
        self.val_query_dir = osp.join(self.dataset_dir, 'val-set-query')
        self.val_query_list = osp.join(self.dataset_dir, 'val-set-query-bbox.txt')
        self.val_gallery_dir = osp.join(self.dataset_dir, 'val-set-gallery')
        self.val_gallery_list = osp.join(self.dataset_dir, 'val-set-gallery-bbox.txt')
        self.test_query_dir = osp.join(self.dataset_dir, 'test-set-query')
        self.test_query_list = osp.join(self.dataset_dir, 'test-set-query-bbox.txt')
        self.test_gallery_dir = osp.join(self.dataset_dir, 'test-set-gallery')
        self.test_gallery_list = osp.join(self.dataset_dir, 'test-set-gallery-bbox.txt')
        self._check_before_run()

        train_names = self._get_names(self.train_list)
        val_query_names = self._get_names(self.val_query_list)
        val_gallery_names = self._get_names(self.val_gallery_list)
        test_query_names = self._get_names(self.test_query_list)
        test_gallery_names = self._get_names(self.test_gallery_list)

        pid2label, clothes2label, pid2clothes = self.get_pid2label_and_clothes2label(train_names)
        train, num_train_pids, num_train_clothes = self._process_dir(self.train_dir, train_names, clothes2label, pid2label=pid2label)

        pid2label, clothes2label = self.get_pid2label_and_clothes2label(val_query_names, val_gallery_names)
        val_query, num_val_query_pids, num_val_query_clothes  = self._process_dir(self.val_query_dir, val_query_names, clothes2label)
        val_gallery, num_val_gallery_pids, num_val_gallery_clothes = self._process_dir(self.val_gallery_dir, val_gallery_names, clothes2label)
        num_val_pids = len(pid2label)
        num_val_clothes = len(clothes2label)

        pid2label, clothes2label = self.get_pid2label_and_clothes2label(test_query_names, test_gallery_names)
        test_query, num_test_query_pids, num_test_query_clothes = self._process_dir(self.test_query_dir, test_query_names, clothes2label)
        test_gallery, num_test_gallery_pids, num_test_gallery_clothes = self._process_dir(self.test_gallery_dir, test_gallery_names, clothes2label)
        num_test_pids = len(pid2label)
        num_test_clothes = len(clothes2label)

        num_total_pids = num_train_pids + num_val_pids + num_test_pids
        num_total_clothes = num_train_clothes + num_val_clothes + num_test_clothes
        num_total_imgs = len(train) + len(val_query) + len(val_gallery) + len(test_query) + len(test_gallery)

        print("=> DeepChange loaded")
        print("Dataset statistics:")
        print("  --------------------------------------------")
        print("  subset        | # ids | # images | # clothes")
        print("  ----------------------------------------")
        print("  train         | {:5d} | {:8d} | {:9d} ".format(num_train_pids, len(train), num_train_clothes))
        print("  query(val)    | {:5d} | {:8d} | {:9d} ".format(num_val_query_pids, len(val_query), num_val_query_clothes))
        print("  gallery(val)  | {:5d} | {:8d} | {:9d} ".format(num_val_gallery_pids, len(val_gallery), num_val_gallery_clothes))
        print("  query         | {:5d} | {:8d} | {:9d} ".format(num_test_query_pids, len(test_query), num_test_query_clothes))
        print("  gallery       | {:5d} | {:8d} | {:9d} ".format(num_test_gallery_pids, len(test_gallery), num_test_gallery_clothes))
        print("  --------------------------------------------")
        print("  total         | {:5d} | {:8d} | {:9d} ".format(num_total_pids, num_total_imgs, num_total_clothes))
        print("  --------------------------------------------")

        self.train = train
        self.val_query = val_query
        self.val_gallery = val_gallery
        self.query = test_query
        self.gallery = test_gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

        self.num_test_pids = num_test_pids
        self.num_test_clothes = num_test_clothes

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
        if not osp.exists(self.val_query_dir):
            raise RuntimeError("'{}' is not available".format(self.val_query_dir))
        if not osp.exists(self.val_gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.val_gallery_dir))
        if not osp.exists(self.test_query_dir):
            raise RuntimeError("'{}' is not available".format(self.test_query_dir))
        if not osp.exists(self.test_gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.test_gallery_dir))

    def _process_dir(self, home_dir, img_names, clothes2label, pid2label=None):
        dataset = []
        pid_container = set()
        clothes_container = set()
        for img_name in img_names:
            img_path = osp.join(home_dir, img_name.split(',')[0])
            names = img_name.split('.')[0].split('_')
            tracklet_id = int(img_name.split(',')[1])
            clothes = names[0] + names[2]
            clothes_id = clothes2label[clothes]
            clothes_container.add(clothes_id)
            pid = int(names[0][1:])
            pid_container.add(pid)
            camid = int(names[1][1:])
            if pid2label is not None:
                pid = pid2label[pid]
            # on DeepChange, we allow the true matches coming from the same camera 
            # but different tracklets as query following the original paper.
            # So we use tracklet_id to replace camid for each sample.
            dataset.append((img_path, pid, tracklet_id, clothes_id))
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        return dataset, num_pids, num_clothes
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/ssd2/data/body/personReID/')
    parser.add_argument('--dataset_name', type=str, default='deepchange_test')
    parser.add_argument('--save_dir', type=str, default='/ssd2/data/body/validation_sets')
    args = parser.parse_args()

    dataset = DeepChange(root=args.dataset_path)



    # Print some statistics
    print(f"Number of identities: {dataset.num_test_pids}")
    print(f"Number of query images: {len(dataset.query)}")
    print(f"Number of gallery images: {len(dataset.gallery)}")
    print(f"Number of clothes: {dataset.num_test_clothes}")



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
        'num_clothes': dataset.num_test_clothes,
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
