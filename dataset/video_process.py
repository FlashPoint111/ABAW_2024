import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms

from utils.randaugment import RandomAugment


class IterCreator(IterableDataset):
    def __init__(self, configs, transform):
        super().__init__()
        self.configs = configs
        self.transform = transform
        self.data_dir = configs["dataset_path"]
        self.data_list = sorted(os.listdir(self.data_dir))
        self.max_len = len(self.data_list)
        self.data_idx = np.random.permutation(self.max_len)

    def __iter__(self):
        for idx in self.data_idx:
            sample = torch.load(os.path.join(self.data_dir, self.data_list[idx]))
            sample["video"] = list(map(lambda x: img_process(x, self.transform), sample["video"]))
            sample["video"] = torch.stack(sample["video"], dim=0).permute(1, 0, 2, 3)
            sample["label"] = np.array(sample["label"], dtype=np.int8)
            sample["audio"] = sample["audio"][:, np.random.randint(5):]
            yield sample

    def __len__(self):
        return self.max_len


def img_process(img: Image, transform: transforms.Compose):
    return transform(img)


def custom_collate_fn(batch):
    videos = torch.stack([item['video'] for item in batch], dim=0)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.int8)
    audios = [torch.tensor(item['audio']) for item in batch]
    audios = pad_sequence(audios, dim=1, padding_value=0)
    return {'video': videos, 'label': labels, 'audio': audios}


def pad_sequence(sequences, dim, padding_value=0.0):
    max_len = max([s.size(dim) for s in sequences])
    out_dims = (len(sequences), sequences[0].size(0), max_len)
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(dim)
        out_tensor[i, ..., :length] = tensor
    return out_tensor


if __name__ == '__main__':

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    with open('../config/train.json') as f:
        configs = json.load(f)
    dataset = IterCreator(configs, pretrain_transform)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn)

    for data in dataloader:
        print(len(data))
