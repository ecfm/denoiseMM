import pickle
import re
import sys

import numpy as np
import torch
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence

from Multimodal_dataset import MultimodalSubdata
from consts import global_consts as gc

if gc.SDK_PATH is None:
    print("SDK path is not specified! Please specify first in constants/paths.py")
    exit(0)
else:
    print("Added gc.SDK_PATH")
    import os

    print(os.getcwd())
    sys.path.append(gc.SDK_PATH)

from mmsdk import mmdatasdk as md

from consts import global_consts as gc

DATASET = md.cmu_mosei

DATA_PATH = gc.data_path

try:
    md.mmdataset(DATASET.highlevel, DATA_PATH)
except RuntimeError:
    print("High-level features have been downloaded previously.")

try:
    md.mmdataset(DATASET.labels, DATA_PATH)
except RuntimeError:
    print("Labels have been downloaded previously.")


class MoseiEmotionDataset(Data.Dataset):
    trainset = MultimodalSubdata("train")
    testset = MultimodalSubdata("test")
    validset = MultimodalSubdata("valid")

    def __init__(self, root, cls="train"):
        self.root = root
        self.cls = cls
        if len(MoseiEmotionDataset.trainset.y) != 0 and cls != "train":
            print("Data has been previously loaded, fetching from previous lists.")
        else:
            self.load_data()

        if self.cls == "train":
            self.dataset = MoseiEmotionDataset.trainset
        elif self.cls == "test":
            self.dataset = MoseiEmotionDataset.testset
        elif self.cls == "valid":
            self.dataset = MoseiEmotionDataset.validset

        self.text = self.dataset.text
        self.audio = self.dataset.audio
        self.vision = self.dataset.vision
        self.y = self.dataset.y

    def multi_collate(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)

        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch])
        visual = None
        acoustic = None
        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
        return sentences, visual, acoustic, labels, lengths

    def load_data(self):

        # obtain the train/dev/test splits - these splits are based on video IDs
        train_split = DATASET.standard_folds.standard_train_fold
        dev_split = DATASET.standard_folds.standard_valid_fold
        test_split = DATASET.standard_folds.standard_test_fold

        dataset = pickle.load(open(os.path.join(gc.data_path, 'mosei_aligned.pkl'), 'rb'))
        # place holders for the final train/dev/test dataset
        data = {'train': {'text': [], 'labels': []}, 'valid': {'text': [], 'labels': []},
                'test': {'text': [], 'labels': []}}
        text_field = 'CMU_MOSEI_TimestampedGloveVectors'
        label_field = 'CMU_MOSEI_LabelsEmotions'
        gc.padding_len = 50

        # define a regular expression to extract the video ID out of the keys
        pattern = re.compile('(.*)\[.*\]')

        for segment in dataset[label_field].keys():

            # get the video ID and the features out of the aligned dataset
            vid = re.search(pattern, segment).group(1)
            label = dataset[label_field][segment]['features']

            # remove nan values
            label = np.nan_to_num(label)
            raw_words = np.asarray(dataset[text_field][segment]['features'])
            if gc.dim_l == -1:
                gc.dim_l = dataset[text_field][segment]['features'].shape[1]
            words = np.zeros((gc.padding_len, gc.dim_l))
            seq_len = min(gc.padding_len, raw_words.shape[0])

            words[:seq_len, :] = raw_words[:seq_len, :]
            split = None
            if vid in train_split:
                split = data['train']
            elif vid in dev_split:
                split = data['valid']
            elif vid in test_split:
                split = data['test']
            # else:
            #     print(f"Found video that doesn't belong to any splits: {vid}")
            if split is not None:
                split['text'].append(words)
                split['labels'].append(label)

        for ds, split_type in [(MoseiEmotionDataset.trainset, 'train'), (MoseiEmotionDataset.validset, 'valid'),
                               (MoseiEmotionDataset.testset, 'test')]:
            ds.text = torch.tensor(np.array(data[split_type]['text']).astype(np.float32)).cpu().detach()
            ds.y = torch.tensor(np.array(data[split_type]['labels']).astype(np.float32)).cpu().detach()


    def __getitem__(self, index):
        return self.text[index], -1, -1, -1, self.y[index].squeeze()

    def __len__(self):
        return len(self.y)
