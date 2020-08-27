import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd


class MultimodalSubdata():
    def __init__(self, name="train"):
        self.name = name
        self.text = np.empty(0)
        self.audio = np.empty(0)
        self.vision = np.empty(0)
        self.y = np.empty(0)


class MultimodalSentiDataset(Data.Dataset):
    trainset = MultimodalSubdata("train")
    testset = MultimodalSubdata("test")
    validset = MultimodalSubdata("valid")
    padding_len = - 1
    dim_l = -1
    dim_a = -1
    dim_v = -1
    output_dim = 1

    def __init__(self, data_path, cls="train"):
        self.data_path = data_path
        self.cls = cls
        if len(MultimodalSentiDataset.trainset.y) != 0 and cls != "train":
            print("Data has been previously loaded, fetching from previous lists.")
        else:
            self.load_data()

        if self.cls == "train":
            self.dataset = MultimodalSentiDataset.trainset
        elif self.cls == "test":
            self.dataset = MultimodalSentiDataset.testset
        elif self.cls == "valid":
            self.dataset = MultimodalSentiDataset.validset

        self.text = self.dataset.text
        self.audio = self.dataset.audio
        self.vision = self.dataset.vision
        self.y = self.dataset.y

    @staticmethod
    def get_loss():
        return nn.MSELoss()

    @staticmethod
    def eval(outputs, labels):
        preds = np.array(outputs)
        truth = np.array(labels)
        mae = np.mean(np.abs(truth - preds))
        return {'mae': mae}

    @staticmethod
    def is_better_metric(current_metric, old_metric):
        if old_metric is None:
            return True
        return current_metric['mae'] < old_metric['mae']

    @staticmethod
    def get_best_metrics(metrics):
        metrics_df = pd.DataFrame(metrics)
        return {'mae': metrics_df['mae'].min()}

    def load_data(self):
        dataset = pickle.load(open(self.data_path, 'rb'))
        MultimodalSentiDataset.padding_len = dataset['test']['text'].shape[1]
        MultimodalSentiDataset.dim_l = dataset['test']['text'].shape[2]
        MultimodalSentiDataset.dim_a = dataset['test']['audio'].shape[2]
        MultimodalSentiDataset.dim_v = dataset['test']['vision'].shape[2]

        for ds, split_type in [(MultimodalSentiDataset.trainset, 'train'), (MultimodalSentiDataset.validset, 'valid'),
                               (MultimodalSentiDataset.testset, 'test')]:
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
