from PIL import Image
from reader import SplittedRecordReader
from general_utils.os_utils import get_all_files
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import Dataset, DataLoader
import os

class RecordDatasetFromList(Dataset):
    def __init__(self, reader, transform, img_list=None):
        super(RecordDatasetFromList, self).__init__()
        if img_list is None:
            img_list = reader.paths
        self.img_list = img_list
        self.reader = reader
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        try:
            result = self.reader.read_by_path(self.img_list[idx])
            if len(result) == 2:
                img, path = result
                crop_kp = None
            else:
                img, path, crop_kp = result

            if img is None:
                raise ValueError(f"Image at path {self.img_list[idx]} could not be loaded.")

            img = img[:, :, :3]  # Ensure image has 3 channels
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, idx, path, crop_kp

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return -1, idx, -1, -1


def make_dataset(data_root, parquet_num_start, parquet_num_end, batch_size, num_workers):
    paths = get_all_files(data_root, extension_list=['.rec'], sort=True)
    print('Found records:', len(paths))
    if parquet_num_end != -1:
        print('parquet_num_start', parquet_num_start)
        print('parquet_num_end', parquet_num_end)
        assert parquet_num_end > parquet_num_start
        in_texts = []
        for i in range(parquet_num_start, parquet_num_end+1):
            in_text = f'/{i}_parquet'
            in_texts.append(in_text)
    else:
        print('parquet_num_start', parquet_num_start)
        print('parquet_num_end', parquet_num_end)
        in_texts = [f'/{parquet_num_start}_parquet']

    paths = [os.path.dirname(path) for path in paths if any(in_text in path for in_text in in_texts)]
    print('Will Process records:', len(paths))
    print('paths', paths)
    reader = SplittedRecordReader(paths)
    transform = Compose([ToTensor(),
                         Resize((112, 112), antialias=True),
                         Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
    dataset = RecordDatasetFromList(reader, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloader
