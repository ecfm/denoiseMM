import pickle

import numpy as np

if __name__ == "__main__":
    dataset = pickle.load(open('mosei_senti_data.pkl', 'rb'))
    test_size = dataset['test']['text'].shape[0]
    padding_len = dataset['test']['text'].shape[1]
    orig_test_txt = np.copy(dataset['test']['text'])

    for mask_ratio in [0.2, 0.4, 0.6]:
        m = np.random.choice([0, 1], p=[mask_ratio, 1-mask_ratio], size=(test_size, padding_len, 1))
        dataset['test']['text'] = orig_test_txt * m
        pickle.dump(dataset['test'], open('mosei_senti_%.0E_mask_data.pkl' % mask_ratio, 'wb'))
