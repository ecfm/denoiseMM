import csv
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
import torch.utils.data as Data
from tqdm import tqdm
import itertools

RAW_TEXT = 'raw_text'
sp_vec = np.array([0.32, 0.25, 0.01, -0.08, -0.31, 0.12, -0.35, 0.07, -0.59, 0.07, -0.05, 0.44, 0.60
, -0.06, 0.20, -0.32, -0.48, 0.34, -0.70, 0.06, 0.11, 0.06, 0.24, 0.09, 0.63, 0.50, 0.35
, 0.22, -0.43, -0.58, 0.27, -0.15, 0.38, 0.21, 0.14, -0.16, 0.51, 0.19, -0.05, -0.09
, -0.30, 0.39, -0.25, -0.45, -0.17, -0.38, 0.64, 0.17, 0.16, 0.12, 0.32, -0.14, -0.44
, -0.30, 0.17, -0.51, 0.71, -0.04, 0.03, -0.07, 0.16, 0.03, -0.25, 0.56, -0.03, -0.17
, 0.03, -0.20, 0.18, -0.30, -0.33, 0.63, -0.28, -0.09, -0.62, -0.44, -0.13, 0.14, 0.10
, 0.23, 0.04, 0.26, 0.27, -0.24, 0.41, 0.42, -0.00, 0.19, -0.48, -0.32, -0.07, -0.35
, 0.17, -0.18, 0.82, 0.34, -0.15, 0.21, 0.11, -0.08, -0.20, 0.31, -0.09, 0.21, 0.32
, -0.08, -0.32, -0.05, -0.54, 0.10, 0.05, -0.35, 0.28, 0.28, -0.14, -0.29, -0.04, 1.05
, 0.02, -0.03, -0.01, -0.49, 0.08, -0.37, 0.24, 0.36, -0.58, 0.35, -0.32, -0.05, -0.50
, -0.42, 0.05, 0.19, 0.69, 0.16, 0.55, -0.34, 1.11, 0.44, 0.14, 0.12, -0.10, 0.24, 0.10
, 0.57, -0.62, 0.20, -0.25, 0.03, 0.39, 0.50, 0.50, 0.38, 0.35, -0.05, 0.88, 0.12, 0.08
, 0.12, -0.58, -0.02, -0.53, 0.21, -0.46, 0.39, -0.06, 0.23, -0.41, -0.21, 0.24, -0.22
, 0.24, 0.74, 0.52, -0.33, 0.36, 0.04, 0.28, 0.04, 0.58, -0.01, -0.52, 0.05, -0.05
, -0.33, 0.18, 0.51, 0.74, 0.57, -0.07, -0.22, -0.11, 0.47, -0.29, -0.57, -0.19, 0.39
, -0.16, 0.19, -0.46, -0.16, -0.27, 0.68, -0.44, 0.01, -0.45, -0.14, 0.67, -0.20, -0.47
, -0.14, -0.09, 0.12, -0.28, 0.24, -0.17, 0.80, 0.53, 0.13, -0.57, -0.57, -0.04, 0.10
, -0.36, -0.27, 0.36, -0.57, -0.27, 0.68, 0.19, -0.10, 0.01, 0.46, 0.62, 0.29, 0.38
, 0.32, 0.63, -0.15, 0.06, -0.48, 0.13, 0.13, -0.35, 0.39, 0.34, 0.25, 0.13, 0.02, -0.07
, -0.51, -0.32, 0.12, -0.32, -0.31, -0.43, 0.71, 0.67, -0.05, 0.16, -0.24, 0.15, 0.14
, 0.04, -0.69, 0.31, 0.81, 0.02, 0.31, 0.04, -1.08, 0.02, 0.01, 0.06, 0.42, 0.53, -0.65
, -0.13, -0.11, -0.75, 0.26, -0.94, -0.66, 0.01, -0.22, -0.18, 0.44, -0.11, -0.87, 0.25
, 0.31, -0.06, 0.36, -0.11, 0.30, 0.07, -1.09, -0.27, -0.01])

pad_vec = np.zeros(300, dtype=float)

def loadGloveModel(gloveFile, vocab):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = defaultdict(lambda: np.random.random(300))
    for line in tqdm(f):
        splitLine = line.split(' ')
        word = splitLine[0]
        if word in vocab:
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print("Done.", len(model)," words loaded!")
    return model

if __name__ == "__main__":
    dataset = pickle.load(open(sys.argv[1], 'rb'))
    vocab = set()
    for split in ['train', 'valid', 'test']:
        vocab.update(itertools.chain(*dataset[split][RAW_TEXT]))
    dic = loadGloveModel(sys.argv[2], vocab)
    dic['<PAD>'] = pad_vec
    dic['sp'] = sp_vec
    for split in ['train', 'valid', 'test']:
        text_embed = []
        for sent in tqdm(dataset[split][RAW_TEXT]):
            sent_embed = [dic[w] for w in sent]
            text_embed.append(sent_embed)
        dataset[split]['text'] = np.array(text_embed)
    pickle.dump(dataset, open(sys.argv[1].split('.pkl')[0]+"_glove.pkl", 'wb'))
