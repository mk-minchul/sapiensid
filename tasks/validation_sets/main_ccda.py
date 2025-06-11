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

import re

class CCDA(object):
    """ DeepChange

    Reference:
        Xu et al. DeepChange: A Long-Term Person Re-Identification Benchmark. arXiv:2105.14685, 2021.

    URL: https://github.com/PengBoXiangShang/deepchange
    """
    dataset_dir = 'CCDA' #Celeb-reID
    def __init__(self, root='', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self._check_before_run()

        train, num_train_pids = self._process_dir(self.train_dir)
        query, num_query_pids  = self._process_dir(self.query_dir)
        gallery, num_gallery_pids = self._process_dir(self.gallery_dir)

        num_total_pids = num_train_pids + num_query_pids + num_gallery_pids
        num_total_imgs = len(train) + len(query) + len(gallery)

        print("=> CCDA loaded")
        print("Dataset statistics:")
        print("  --------------------------------------------")
        print("  subset        | # ids | # images")
        print("  ----------------------------------------")
        print("  train         | {:5d} | {:8d}  ".format(num_train_pids, len(train)))
        print("  query         | {:5d} | {:8d}  ".format(num_query_pids, len(query)))
        print("  gallery       | {:5d} | {:8d}  ".format(num_gallery_pids, len(gallery)))
        print("  --------------------------------------------")
        print("  total         | {:5d} | {:8d}  ".format(num_total_pids, num_total_imgs))
        print("  --------------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_test_pids = num_query_pids
        self.num_train_clothes = 10

        self.pid2clothes = np.ones((self.num_train_pids, self.num_train_clothes))


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, home_dir, relabel=True):
  
        pattern = re.compile(r'([-\d]+)_(\d)')
        fpaths = sorted(glob.glob(osp.join(home_dir, '*.jpg')))
        i = 0
        dataset = []
        pid_container = set()
        all_pids = {}
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored

            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            cam -= 1
            clothes_id = 1 #-1
            pid = all_pids[pid]
            pid_container.add(pid)
            i = i+1
            dataset.append((fpath, pid, cam, clothes_id))
        num_pids = len(pid_container)

        return dataset, num_pids


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/ssd2/data/body/personReID/')
    parser.add_argument('--dataset_name', type=str, default='ccda_test')
    parser.add_argument('--save_dir', type=str, default='/ssd2/data/body/validation_sets')
    args = parser.parse_args()

    dataset = CCDA(root=args.dataset_path)
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
