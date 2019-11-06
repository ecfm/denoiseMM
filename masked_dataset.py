import os
import pickle

import numpy as np
import torch
import torch.utils.data as Data

from consts import global_consts as gc


class MultimodalSubdata():
    def __init__(self, name="train"):
        self.name = name
        self.text = np.empty(0)
        self.audio = np.empty(0)
        self.vision = np.empty(0)
        self.y = np.empty(0)


class MaskedDataset(Data.Dataset):
    testset = MultimodalSubdata("test")

    def __init__(self, root, dataset_name, cls="test"):
        self.root = root
        self.cls = cls
        self.load_data(dataset_name)

        self.dataset = MaskedDataset.testset

        self.text = self.dataset.text
        self.audio = self.dataset.audio
        self.vision = self.dataset.vision
        self.y = self.dataset.y


    def load_data(self, dataset_name):
        dataset_path = os.path.join(gc.data_path, dataset_name)
        dataset = pickle.load(open(dataset_path, 'rb'))
        gc.padding_len = dataset['text'].shape[1]
        gc.dim_l = dataset['text'].shape[2]
        gc.dim_a = dataset['audio'].shape[2]
        gc.dim_v = dataset['vision'].shape[2]

        for ds, split_type in [(MaskedDataset.testset, 'test')]:
            ds.text = torch.tensor(dataset['text'].astype(np.float32)).cpu().detach()
            ds.audio = torch.tensor(dataset['audio'].astype(np.float32))
            ds.audio[ds.audio == -np.inf] = 0
            ds.audio = ds.audio.clone().cpu().detach()
            ds.vision = torch.tensor(dataset['vision'].astype(np.float32)).cpu().detach()
            ds.y = torch.tensor(dataset['labels'].astype(np.float32)).cpu().detach()

    def __getitem__(self, index):
        inputLen = len(self.text[index])
        return self.text[index], self.audio[index], self.vision[index], \
               inputLen, self.y[index].squeeze()

    def __len__(self):
        return len(self.y)

