import json
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, f1_score

from masked_dataset import MaskedDataset
from consts import global_consts as gc
from model import Net

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8
}


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def eval_mosei_emo(split, output_all, label_all):
    truths = np.array(label_all).reshape((-1, len(gc.best.mosei_cls)))
    preds = np.array(output_all).reshape((-1, len(gc.best.mosei_cls)))
    cls_mae = {}
    for cls_id, cls in enumerate(gc.best.mosei_cls):
        mae = np.mean(np.absolute(preds[:, cls_id] - truths[:, cls_id]))
        cls_mae[cls] = mae
        print("\t%s %s mae: %f" % (split, cls, round(mae, 3)))
    return cls_mae

def eval_senti(split, output_all, label_all):
    truth = np.array(label_all)
    preds = np.array(output_all)
    mae = np.mean(np.abs(truth - preds))
    acc = accuracy_score(truth >= 0, preds >= 0)
    corr = np.corrcoef(preds, truth)[0][1]
    non_zeros = np.array([i for i, e in enumerate(truth) if e != 0])

    preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    truth_a7 = np.clip(truth, a_min=-3., a_max=3.)
    preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
    truth_a5 = np.clip(truth, a_min=-2., a_max=2.)
    acc_7 = multiclass_acc(preds_a7, truth_a7)
    acc_5 = multiclass_acc(preds_a5, truth_a5)
    f1_mfn = f1_score(np.round(truth), np.round(preds), average="weighted")
    f1_raven = f1_score(truth >= 0, preds >= 0, average="weighted")
    f1_muit = f1_score((preds[non_zeros] > 0), (truth[non_zeros] > 0), average='weighted')
    binary_truth = (truth[non_zeros] > 0)
    binary_preds = (preds[non_zeros] > 0)
    ex_zero_acc = accuracy_score(binary_truth, binary_preds)
    print("\t%s mean error: %f" % (split, mae))
    # print("\t%s correlation coefficient: %f" % (split, corr))
    # print("\t%s accuracy: %f" % (split, acc))
    # print("\t%s mult_acc_7: %f" % (split, acc_7))
    # print("\t%s mult_acc_5: %f" % (split, acc_5))
    # print("\t%s F1 MFN: %f " % (split, f1_mfn))
    # print("\t%s F1 RAVEN: %f " % (split, f1_raven))
    # print("\t%s F1 MuIT: %f " % (split, f1_muit))
    # print("\t%s exclude zero accuracy: %f" % (split, ex_zero_acc))
    return mae, corr, acc, acc_7, acc_5, f1_mfn, f1_raven, f1_muit, ex_zero_acc


if __name__ == "__main__":
    for mask_ratio in [0.2, 0.4, 0.6]:
        ds = MaskedDataset
        conf = sys.argv[1]
        gc.config = json.load(open("configs/%s.json" % conf), object_pairs_hook=OrderedDict)
        test_dataset = ds(gc.data_path, 'mosei_senti_%.0E_mask_data.pkl' % mask_ratio, cls="test")
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=1,
        )

        if gc.single_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
        else:
            device = torch.device("cuda:%d" % gc.config['cuda'] if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                torch.cuda.set_device(gc.config['cuda'])
        gc.device = device
        print("running device: ", device)

        net = Net()
        checkpoint = torch.load("model/mosei_senti_%s.tar" % conf, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.to(device)

        with torch.no_grad():
            test_label_all = []
            test_output_all = []
            for data in test_loader:
                words, covarep, facet, inputLen, labels = data
                if covarep.size()[0] == 1:
                    continue
                words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(
                    device), inputLen.to(device), labels.to(device)
                outputs = net(words, covarep, facet, inputLen)

                test_output_all.extend(outputs.squeeze().tolist())
                test_label_all.extend(labels.tolist())

            best_model = False

            if gc.dataset == 'mosei_emo':
                test_mae = eval_mosei_emo('test', test_output_all, test_label_all)
                for cls in gc.best.mosei_cls:
                    if test_mae[cls] < gc.best.mosei_emo_best_mae[cls]:
                        gc.best.mosei_emo_best_mae[cls] = test_mae[cls]
                        best_model = True
            else:
                test_mae, test_cor, test_acc, test_acc_7, test_acc_5, test_f1_mfn, test_f1_raven, test_f1_muit, \
                test_ex_zero_acc = eval_senti('test', test_output_all, test_label_all)

    # logSummary()