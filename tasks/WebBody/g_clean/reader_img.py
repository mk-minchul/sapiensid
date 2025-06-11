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


class RecordReader():

    def __init__(self, root='/mckim/temp/temp_recfiles', prefix='train', return_img=True):
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
        self.return_img = return_img
        self.face_confidence = pd.read_csv(os.path.join(root, f'face_confidence.csv'))


    def dispose(self):
        self.record.close()

    def read_by_index(self, index):
        if self.return_img:
            header, binary = mx.recordio.unpack(self.record.read_idx(index))
            image = mx.image.imdecode(binary).asnumpy()
            path = self.index_to_path[index]
            face_confidence = self.face_confidence.loc[index, 'confidence']
            return image, path, face_confidence
        else:
            path = self.index_to_path[index]
            image = None
            face_confidence = self.face_confidence.loc[index, 'confidence']
            return image, path, face_confidence


    def read_by_path(self, path):
        index = self.path_to_index[path]
        return self.read_by_index(index)

    def export(self, save_root):
        for idx in self.index_to_path.keys():
            image, path = self.read_by_index(idx)
            img_save_path = os.path.join(save_root, path)
            os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
            cv2.imwrite(img_save_path, image)


class SplittedRecordReader():
    def __init__(self, roots, return_img=True):
        print(f'Loading {len(roots)} records')
        print(roots)
        self.records = []
        for root in tqdm(roots, total=len(roots), desc='Loading records'):
            self.records.append(RecordReader(root, return_img=return_img))
        self.path_to_record_num = {}
        for record_idx, record in enumerate(self.records):
            for key in record.path_to_index.keys():
                self.path_to_record_num[key] = record_idx
        self.paths = list(self.path_to_record_num.keys())
        self.return_img = return_img

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

