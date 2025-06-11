from PIL import Image
from reader import SplittedRecordReader
from general_utils.os_utils import get_all_files
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

class RecordDatasetFromList(Dataset):
    def __init__(self, reader, transform, img_list=None, path_to_label=None):
        super(RecordDatasetFromList, self).__init__()
        if img_list is None:
            img_list = reader.paths
        self.img_list = img_list

        # sort img_list according to reader.paths to read faster in hdd
        img_list = set(img_list)
        reader_paths = pd.Series(reader.paths)
        img_list = reader_paths[reader_paths.apply(lambda x: x in img_list)].tolist()
        assert set(self.img_list) == set(img_list)
        self.img_list = img_list

        self.reader = reader
        self.transform = transform
        self.path_to_label = path_to_label

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        try:
            result = self.reader.read_by_path(self.img_list[idx])
            img, path, crop_kp, orig_shape = result

            if img is None:
                raise ValueError(f"Image at path {self.img_list[idx]} could not be loaded.")

            img = img[:, :, :3]  # Ensure image has 3 channels
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            label = self.path_to_label[path]
            return img, idx, path, crop_kp, label, orig_shape

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return -1, idx, -1, -1, -1, -1


def make_dataset(paths, parquet_num_start, parquet_num_end, path_to_label, image_size=384):

    print('Found records:', len(paths))
    if parquet_num_end != -1:
        print('parquet_num_start', parquet_num_start)
        print('parquet_num_end', parquet_num_end)
        assert parquet_num_end > parquet_num_start
        in_texts = []
        for i in range(parquet_num_start, parquet_num_end+1):
            in_text = f'/{i}_parquet'
            in_texts.append(in_text)
        paths = [os.path.dirname(path) for path in paths if any(in_text in path for in_text in in_texts)]
    else:
        paths = [os.path.dirname(path) for path in paths]
    
    print('Will Process records:', len(paths))
    print('paths', paths)
    reader = SplittedRecordReader(paths, image_size=image_size)
    img_list = list(path_to_label.keys())
    print('intended img_list', len(img_list))
    interesecting_img_list = list(set(reader.paths).intersection(set(img_list)))
    if len(interesecting_img_list) != len(img_list):
        img_list = interesecting_img_list
    else:
        img_list = img_list
    print('actual img_list', len(img_list))
    dataset = RecordDatasetFromList(reader, transform=None, img_list=img_list, path_to_label=path_to_label)

    return dataset
