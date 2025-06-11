from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd
import mxnet as mx
import cv2
import atexit
import numpy as np
import torch


class FeatureRecordReader():

    def __init__(self, root='/mckim/temp/temp_recfiles', prefix='train'):
        path_imgidx = os.path.join(root, f'{prefix}.idx')
        path_imgrec = os.path.join(root, f'{prefix}.rec')
        self.record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        path_list = os.path.join(root, f'{prefix}.tsv')
        info = pd.read_csv(path_list, sep='\t', index_col=0, header=None)
        self.index_to_path = dict(info[1])
        self.path_to_index = {v:k for k,v in self.index_to_path.items()}
        self.info = info
        self.info.columns = ['path', 'label']
        atexit.register(self.dispose)

        if os.path.isfile(os.path.join(root, f'face_confidence.csv')):
            self.load_aux = True
            self.face_confidence = pd.read_csv(os.path.join(root, f'face_confidence.csv'))
            self.face_kps_xyn = pd.read_csv(os.path.join(root, f'face_kps_xyn.csv'))
            self.keypoint_cols = np.array([[f'kp_{k}_x', f'kp_{k}_y'] for k in range(5)]).flatten()
        else:
            self.load_aux = False


    def dispose(self):
        self.record.close()

    def read_by_index(self, index):
        header, binary = mx.recordio.unpack(self.record.read_idx(index))
        label = header.label
        label = torch.tensor(label, dtype=torch.long)
        sample = np.frombuffer(binary, dtype=np.float16)
        sample = np.array(sample)
        sample = torch.from_numpy(sample).float()
        path = self.index_to_path[index]
        return sample, label, path



    def read_by_path(self, path):
        index = self.path_to_index[path]
        return self.read_by_index(index)

    def export(self, save_root):
        for idx in self.index_to_path.keys():
            image, path = self.read_by_index(idx)
            img_save_path = os.path.join(save_root, path)
            os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
            cv2.imwrite(img_save_path, image)


class SplittedFeatureRecordReader():
    def __init__(self, roots):
        print(f'Loading {len(roots)} records')
        print(roots)
        self.records = []
        for root in tqdm(roots, total=len(roots), desc='Loading records'):
            self.records.append(FeatureRecordReader(root))
        self.path_to_record_num = {}
        for record_idx, record in enumerate(self.records):
            for key in record.path_to_index.keys():
                self.path_to_record_num[key] = record_idx
        self.paths = list(self.path_to_record_num.keys())

    def __len__(self):
        return len(self.paths)

    def read_by_index(self, index):
        return self.read_by_path(self.paths[index])

    def read_by_path(self, path):
        record_num = self.path_to_record_num[path]
        return self.records[record_num].read_by_path(path)

    def export(self, save_root):
        raise NotImplementedError('')

    def existing_keys(self):
        return self.path_to_record_num.keys()

    def load_done_list(self):
        donelist = set()
        for record in self.records:
            _donelist = record.load_done_list()
            if _donelist is not None:
                donelist = donelist | _donelist
        return donelist

