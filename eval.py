import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data as Data

from consts import global_consts as gc
from masked_dataset import MaskedDataset
from net import Net

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8
}

def eval_senti(split, mod, output_all, label_all):
    truth = np.array(label_all)
    preds = np.array(output_all)
    mae = np.mean(np.abs(truth - preds))
    # acc = accuracy_score(truth >= 0, preds >= 0)
    # corr = np.corrcoef(preds, truth)[0][1]
    # non_zeros = np.array([i for i, e in enumerate(truth) if e != 0])
    #
    # preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    # truth_a7 = np.clip(truth, a_min=-3., a_max=3.)
    # preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
    # truth_a5 = np.clip(truth, a_min=-2., a_max=2.)
    # acc_7 = multiclass_acc(preds_a7, truth_a7)
    # acc_5 = multiclass_acc(preds_a5, truth_a5)
    # f1_mfn = f1_score(np.round(truth), np.round(preds), average="weighted")
    # f1_raven = f1_score(truth >= 0, preds >= 0, average="weighted")
    # f1_muit = f1_score((preds[non_zeros] > 0), (truth[non_zeros] > 0), average='weighted')
    # binary_truth = (truth[non_zeros] > 0)
    # binary_preds = (preds[non_zeros] > 0)
    # ex_zero_acc = accuracy_score(binary_truth, binary_preds)
    # print("\t%s mae_%s : %f" % (split, mod, mae))
    return mae


def get_test_metrics(epoch, device, test_loader, net):
    with torch.no_grad():
        test_label_all = []
        test_output_l_all = []
        test_output_all = []

        for data in test_loader:
            words, covarep, facet, inputLen, labels = data
            words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(device), \
                                                      inputLen.to(device), labels.to(device)
            if covarep.size()[0] == 1:
                continue
            outputs_l, outputs = net(x_l=words, x_a=covarep, x_v=facet)
            test_output_all.extend(outputs.tolist())
            test_output_l_all.extend(outputs_l.tolist())
            test_label_all.extend(labels.tolist())

        best_model = False
        test_mae_l = eval_senti('test', 'l', test_output_l_all, test_label_all)
        test_mae = eval_senti('test', 'lav', test_output_all, test_label_all)
        return best_model, test_mae, test_mae_l


if __name__ == "__main__":
    ds = MaskedDataset
    for config_file_name in os.listdir("configs"):
        config_name = os.path.splitext(os.path.basename(config_file_name))[0]
        model_name = config_name
        model_path = os.path.join(gc.model_path, gc.dataset + '_' + model_name + '.tar')
        if not os.path.exists(model_path):
            continue
        gc.config = json.load(open("configs/%s.json" % config_name), object_pairs_hook=OrderedDict)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        gc.device = device
        net = Net()
        try:
            checkpoint = torch.load(model_path, map_location=device)
            net.load_state_dict(checkpoint['state'])
            net.to(device)
        except:
            continue
        # print("Evaluating model "+model_name)
        maes_l = []
        maes = []
        for mask_ratio in [0.2, 0.4, 0.6]:
            test_dataset = ds(gc.data_path, 'mosei_senti_%.0E_mask_data.pkl' % mask_ratio, cls="test")
            test_loader = Data.DataLoader(
                dataset=test_dataset,
                batch_size=100,
                shuffle=False,
                num_workers=1,
            )
            _, mae, mae_l = get_test_metrics(-1, device, test_loader, net)
            maes.append(mae)
            maes_l.append(mae_l)
        print("%s, lav, %f,%f,%f,%f" % (config_name, gc.best.min_test_mae, maes[0], maes[1], maes[2]))
        with open(os.path.join(gc.model_path, config_name + "_results.csv"), "w") as f:
            f.write("%s, lav, %f,%f,%f,%f\n" % (config_name, gc.best.min_test_mae, maes[0], maes[1], maes[2]))

    # logSummary()