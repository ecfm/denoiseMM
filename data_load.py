import os
import pickle

import numpy as np
import torch
import torch.utils.data as Data

from consts import global_consts as gc


class MultimodalSubdata():
    def __init__(self, name="train"):
        self.name = name
        self.language = np.empty(0)
        self.acoustic = np.empty(0)
        self.visual = np.empty(0)
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

        self.language = self.dataset.language
        self.acoustic = self.dataset.acoustic
        self.visual = self.dataset.visual
        self.y = self.dataset.y


    def load_data(self):
        dataset_path = os.path.join(gc.data_path, gc.dataset + '_data.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'), encoding='latin1')
        gc.padding_len = dataset['test']['language'].shape[1]
        gc.dim_l = dataset['test']['language'].shape[2]
        gc.dim_a = dataset['test']['acoustic'].shape[2]
        gc.dim_v = dataset['test']['visual'].shape[2]

        for ds, split_type in [(MultimodalDataset.trainset, 'train'), (MultimodalDataset.validset, 'valid'),
                               (MultimodalDataset.testset, 'test')]:
            ds.language = torch.tensor(dataset[split_type]['language'].astype(np.float32)).cpu().detach()
            ds.acoustic = torch.tensor(dataset[split_type]['acoustic'].astype(np.float32))
            ds.acoustic[ds.acoustic == -np.inf] = 0
            ds.acoustic = ds.acoustic.clone().cpu().detach()
            ds.visual = torch.tensor(dataset[split_type]['visual'].astype(np.float32)).cpu().detach()
            ds.y = torch.tensor(dataset[split_type]['labels_sent'].astype(np.float32)).cpu().detach()

    def __getitem__(self, index):
        inputLen = len(self.language[index])
        return self.language[index], self.acoustic[index], self.visual[index], \
               inputLen, self.y[index].squeeze()

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    dataset = MultimodalDataset(gc.data_path)
