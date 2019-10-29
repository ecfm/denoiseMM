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


class MultimodalDataset(Data.Dataset):
    trainset = MultimodalSubdata("train")
    testset = MultimodalSubdata("test")
    validset = MultimodalSubdata("valid")

    def __init__(self, root, cls="train"):
        self.root = root
        self.cls = cls
        if len(MultimodalDataset.trainset.y) != 0 and cls != "train":
            print("Data has been previously loaded, fetching from previous lists.")
        else:
            self.load_data()

        if self.cls == "train":
            self.dataset = MultimodalDataset.trainset
        elif self.cls == "test":
            self.dataset = MultimodalDataset.testset
        elif self.cls == "valid":
            self.dataset = MultimodalDataset.validset

        self.text = self.dataset.text
        self.audio = self.dataset.audio
        self.vision = self.dataset.vision
        self.y = self.dataset.y


    def load_data(self):
        dataset_path = os.path.join(gc.data_path, gc.dataset + '_data.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))
        gc.padding_len = dataset['test']['text'].shape[1]
        gc.dim_l = dataset['test']['text'].shape[2]
        gc.dim_a = dataset['test']['audio'].shape[2]
        gc.dim_v = dataset['test']['vision'].shape[2]

        for ds, split_type in [(MultimodalDataset.trainset, 'train'), (MultimodalDataset.validset, 'valid'),
                               (MultimodalDataset.testset, 'test')]:
            ds.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
            ds.audio = torch.tensor(dataset[split_type]['audio'].astype(np.float32))
            ds.audio[ds.audio == -np.inf] = 0
            ds.audio = ds.audio.clone().cpu().detach()
            ds.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
            ds.y = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()

    def __getitem__(self, index):
        inputLen = len(self.text[index])
        return self.text[index], self.audio[index], self.vision[index], \
               inputLen, self.y[index].squeeze()

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    dataset = MultimodalDataset(gc.data_path)
