import csv
import os
import pickle
import re
import sys
from collections import OrderedDict

import numpy as np
import torch.utils.data as Data
from tqdm import tqdm

import sys
sys.path.append('/Users/cask/Dropbox (MIT)/OneDrive/MultiComp/CMU-MultimodalSDK')
from mmsdk import mmdatasdk


sp_vec = """[0.32 0.25 0.01 -0.08 -0.31 0.12 -0.35 0.07 -0.59 0.07 -0.05 0.44 0.60
 -0.06 0.20 -0.32 -0.48 0.34 -0.70 0.06 0.11 0.06 0.24 0.09 0.63 0.50 0.35
 0.22 -0.43 -0.58 0.27 -0.15 0.38 0.21 0.14 -0.16 0.51 0.19 -0.05 -0.09
 -0.30 0.39 -0.25 -0.45 -0.17 -0.38 0.64 0.17 0.16 0.12 0.32 -0.14 -0.44
 -0.30 0.17 -0.51 0.71 -0.04 0.03 -0.07 0.16 0.03 -0.25 0.56 -0.03 -0.17
 0.03 -0.20 0.18 -0.30 -0.33 0.63 -0.28 -0.09 -0.62 -0.44 -0.13 0.14 0.10
 0.23 0.04 0.26 0.27 -0.24 0.41 0.42 -0.00 0.19 -0.48 -0.32 -0.07 -0.35
 0.17 -0.18 0.82 0.34 -0.15 0.21 0.11 -0.08 -0.20 0.31 -0.09 0.21 0.32
 -0.08 -0.32 -0.05 -0.54 0.10 0.05 -0.35 0.28 0.28 -0.14 -0.29 -0.04 1.05
 0.02 -0.03 -0.01 -0.49 0.08 -0.37 0.24 0.36 -0.58 0.35 -0.32 -0.05 -0.50
 -0.42 0.05 0.19 0.69 0.16 0.55 -0.34 1.11 0.44 0.14 0.12 -0.10 0.24 0.10
 0.57 -0.62 0.20 -0.25 0.03 0.39 0.50 0.50 0.38 0.35 -0.05 0.88 0.12 0.08
 0.12 -0.58 -0.02 -0.53 0.21 -0.46 0.39 -0.06 0.23 -0.41 -0.21 0.24 -0.22
 0.24 0.74 0.52 -0.33 0.36 0.04 0.28 0.04 0.58 -0.01 -0.52 0.05 -0.05
 -0.33 0.18 0.51 0.74 0.57 -0.07 -0.22 -0.11 0.47 -0.29 -0.57 -0.19 0.39
 -0.16 0.19 -0.46 -0.16 -0.27 0.68 -0.44 0.01 -0.45 -0.14 0.67 -0.20 -0.47
 -0.14 -0.09 0.12 -0.28 0.24 -0.17 0.80 0.53 0.13 -0.57 -0.57 -0.04 0.10
 -0.36 -0.27 0.36 -0.57 -0.27 0.68 0.19 -0.10 0.01 0.46 0.62 0.29 0.38
 0.32 0.63 -0.15 0.06 -0.48 0.13 0.13 -0.35 0.39 0.34 0.25 0.13 0.02 -0.07
 -0.51 -0.32 0.12 -0.32 -0.31 -0.43 0.71 0.67 -0.05 0.16 -0.24 0.15 0.14
 0.04 -0.69 0.31 0.81 0.02 0.31 0.04 -1.08 0.02 0.01 0.06 0.42 0.53 -0.65
 -0.13 -0.11 -0.75 0.26 -0.94 -0.66 0.01 -0.22 -0.18 0.44 -0.11 -0.87 0.25
 0.31 -0.06 0.36 -0.11 0.30 0.07 -1.09 -0.27 -0.01]"""

def get_words(raw_words_dataset, vid, start_idx, end_idx):
    words = raw_words_dataset[vid]['features'][max(start_idx, 0):end_idx]
    words = words[-50:]
    words = [w[0].decode('utf-8') for w in words]
    sp_idx = np.array([w_i for w_i, w in enumerate(words) if w == 'sp'])
    return words, sp_idx

if __name__ == "__main__":
    file_root = "/Users/cask/Downloads"
    m = pickle.load(open(f"{file_root}/mosei_senti_data.pkl", 'rb'))
    arr_formatter = {'float_kind': '{:0.2f}'.format}
    recipe = {"words": f"{file_root}/CMU_MOSEI_TimestampedWords.csd", }
    # "vecs": f"{file_root}/CMU_MOSEI_TimestampedWordVectors.csd"}
    raw_words_dataset = mmdatasdk.mmdataset(recipe)['words']
    for mode in ['train', 'valid', 'test']:
        vecs = m[mode]['text']
        ids = m[mode]['id']
        m[mode]['raw_text'] = []
        # raw_vecs = mmdatasdk.mmdataset(recipe)['vecs']
        # vec2word = {}
        # vocab = set()
        # for vid in raw_words.keys():
        #     if mmdatasdk.cmu_mosei.cmu_mosei_std_folds.standard_valid_fold:
        #         assert np.array_equal(raw_vecs[vid]['intervals'], raw_words[vid]['intervals'])
        #         _words = raw_words[vid]['features']
        #         words = []
        #         for i, word in enumerate(_words):
        #             w = word[0].decode('utf-8')  # SDK stores strings as bytes, decode into strings here
        #             if w in vocab:
        #                 continue
        #             vocab.add(w)
        #             vec_str = np.array2string(raw_vecs[vid]['features'][i], formatter=arr_formatter)
        #             vec2word[vec_str] = w
        ones = 0
        twos = 0
        my_text = []
        debug_idx = 0
        for i, (vid, start, end) in enumerate(tqdm(ids[debug_idx:]), start=debug_idx):
            raw_int_dataset = np.round(raw_words_dataset[vid]['intervals'], 3)
            start_idx = np.argmax(raw_int_dataset[:, 1] >= np.round(float(start), 3))
            end_idx = np.argmax(raw_int_dataset[:, 0] >= np.round(float(end), 3))
            if end_idx == 0:
                end_idx = len(raw_int_dataset)
            words, sp_idx = get_words(raw_words_dataset, vid, start_idx, end_idx)
            t = []
            vec_start_idx = np.argmax(np.sum(np.abs(vecs[i]), axis=1) > 0)
            vecs_len = 50 - vec_start_idx
            sp_vec_idx = np.array([v_i - vec_start_idx for v_i, v in enumerate(vecs[i])
                          if np.array2string(v, formatter=arr_formatter) == sp_vec])
            if (not (len(sp_vec_idx) > 0 and len(sp_idx) == 0))\
                    and (not np.array_equal(sp_vec_idx, sp_idx)) or vecs_len != len(words):
                len_diff = len(words) - vecs_len
                words, sp_idx = get_words(raw_words_dataset, vid, start_idx+len_diff, end_idx)
                if (not (len(sp_vec_idx) > 0 and len(sp_idx) == 0))\
                    and (not np.array_equal(sp_vec_idx, sp_idx)) or vecs_len != len(words):
                    words, sp_idx = get_words(raw_words_dataset, vid, start_idx, end_idx - len_diff)
                if (not (len(sp_vec_idx) > 0 and len(sp_idx) == 0))\
                    and (not np.array_equal(sp_vec_idx, sp_idx)) or vecs_len != len(words):
                    words, sp_idx = get_words(raw_words_dataset, vid, start_idx - 1, end_idx - 1)
                if (not (len(sp_vec_idx) > 0 and len(sp_idx) == 0))\
                    and (not np.array_equal(sp_vec_idx, sp_idx)) or vecs_len != len(words):
                    words, sp_idx = get_words(raw_words_dataset, vid, start_idx - 1, end_idx + 1)
                if (not (len(sp_vec_idx) > 0 and len(sp_idx) == 0))\
                    and (not np.array_equal(sp_vec_idx, sp_idx)) or vecs_len != len(words):
                    words, sp_idx = get_words(raw_words_dataset, vid, start_idx + 1, end_idx - 1)
                if (not (len(sp_vec_idx) > 0 and len(sp_idx) == 0))\
                    and (not np.array_equal(sp_vec_idx, sp_idx)) or vecs_len != len(words):
                    words, sp_idx = get_words(raw_words_dataset, vid, start_idx + 1, end_idx + 1)
                if (not (len(sp_vec_idx) > 0 and len(sp_idx) == 0))\
                    and (not np.array_equal(sp_vec_idx, sp_idx)) or (vecs_len != len(words) and len(sp_idx) <= 1):
                    print("{} != {} or {} != {}, {}-th id {} in mode {}".format(sp_vec_idx, sp_idx,
                                                                                vecs_len, len(words), i, ids[i], mode))
                    print('>>>> {}'.format(raw_words_dataset[vid]['intervals'][start_idx-1:start_idx + 1]))
                    print('<<<< {}'.format(raw_words_dataset[vid]['intervals'][end_idx - 2:end_idx]))
                    print("     ".join([w if (w_i + 1) % 6 != 0 else w + '\n'
                                              for w_i, w in enumerate(words)]))
                    print("vecs: {}".format(np.sum(np.abs(vecs[i][vec_start_idx:]), axis=1)))
            # assert vecs_len == len(words), "{} != {}, {}-th id {}".format(vecs_len, len(words), i, ids[i])



            # j = 0
            # for e in vecs[i]:
            #     if np.sum(np.abs(e)) < 1e-9:
            #         t.append('<PAD>')
            #         continue
            #     # vec_str = np.array2string(e, formatter=arr_formatter)
            #     word_j = words[j][0].decode('utf-8')
            #     # if vec_str in vec2word:
            #     #     assert word_j == vec2word[vec_str], "{} != {} at {}".format(word_j, vec2word[vec_str], ids[i])
            #     # else:
            #     #     vec2word[vec_str] = word_j
            #     t.append(word_j)
            #     j += 1
            t = ['<PAD>']*vec_start_idx + words
            assert len(t) == 50
            m[mode]['raw_text'].append(t)
            # my_text.append(' '.join(t))
        # print(t)
    # [s for s in my_text if 'UNK_' in ' '.join(s)]
    pickle.dump(my_text, open("mosei_senti_data_text.pkl.pkl", "wb"))
