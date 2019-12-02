import json
import os
import signal
import sys
import time
from collections import OrderedDict

import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, f1_score

from consts import global_consts as gc
from models import MULTModel

lambda_q = 0.15

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8
}
np.random.seed(0)

def stopTraining(signum, frame):
    global savedStdout
    logSummary()
    sys.stdout = savedStdout
    sys.exit()


def train_model(args, config_file_name, model_name):
    save_epochs = [1, 10, 50, 100, 150, 200, 500, 700, 999]
    config_name = ''
    if config_file_name:
        config_name = os.path.splitext(os.path.basename(config_file_name))[0]
    if model_name is None:
        model_name = config_name
    try:
        signal.signal(signal.SIGINT, stopTraining)
        signal.signal(signal.SIGTERM, stopTraining)
    except:
        pass

    global savedStdout
    savedStdout = sys.stdout
    if gc.dataset == 'mosei_emo':
        from MOSEI_emo_dataset import MoseiEmotionDataset
        ds = MoseiEmotionDataset
    else:
        from Multimodal_dataset import MultimodalDataset
        ds = MultimodalDataset

    train_dataset = ds(gc.data_path, cls="train")
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=gc.batch_size,
        shuffle=True,
        num_workers=1,
    )

    test_dataset = ds(gc.data_path, cls="test")
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=gc.batch_size,
        shuffle=False,
        num_workers=1,
    )

    print("HPID:%d:Data Successfully Loaded." % gc.HPID)

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
    gc().logParameters()
    hyp_params = args
    hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = gc.dim_l, gc.dim_a, gc.dim_v
    # hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_dataset.__len__()

    net = MULTModel(output_dim_dict.get(gc.dataset, 1))
    print(net)
    net.to(device)

    if gc.dataset == "iemocap":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=gc.config['lr'])
    start_epoch = 0
    model_path = os.path.join(gc.model_path, gc.dataset + '_' + model_name + '.tar')
    if gc.load_model and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        gc.best = checkpoint['best']

    if gc.dataset == 'mosei_emo':
        eval_method = eval_mosei_emo
    else:
        eval_method = eval_senti

    running_loss = 0.0
    for epoch in range(start_epoch, gc.config['epoch_num']):
        if epoch % 10 == 0:
            print("HPID:%d:Training Epoch %d." % (gc.HPID, epoch))
        if gc.save_grad and epoch in save_epochs:
            grad_dict = {}
            update_dict = {}
        if epoch % 100 == 0:
            logSummary()
        if gc.lr_decay and (epoch == 75 or epoch == 200):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2
        with torch.no_grad():
            print("Epoch #%d results:" % epoch)
            test_label_all = []
            test_output_all = []
            for data in test_loader:
                words, covarep, facet, inputLen, labels = data
                if covarep.size()[0] == 1:
                    continue
                words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(
                    device), inputLen.to(device), labels.to(device)
                outputs, _ = net(words, covarep, facet)

                test_output_all.extend(outputs.tolist())
                test_label_all.extend(labels.tolist())

            best_model = False

            if gc.dataset == 'mosei_emo':
                test_mae = eval_mosei_emo('test', test_output_all, test_label_all)
                for cls in gc.best.mosei_cls:
                    if test_mae[cls] < gc.best.mosei_emo_best_mae[cls]:
                        gc.best.mosei_emo_best_mae[cls] = test_mae[cls]
                        gc.best.best_epoch = epoch + 1
                        best_model = True
            else:
                test_mae, test_cor, test_acc, test_acc_7, test_acc_5, test_f1_mfn, test_f1_raven, test_f1_muit, \
                test_ex_zero_acc = eval_senti('test', test_output_all, test_label_all)
                if len(test_output_all) > 0:
                    if test_mae < gc.best.min_test_mae:
                        gc.best.min_test_mae = test_mae
                        gc.best.best_epoch = epoch + 1
                        best_model = True
                    if test_cor > gc.best.max_test_cor:
                        gc.best.max_test_cor = test_cor
                    if test_acc > gc.best.max_test_acc:
                        gc.best.max_test_acc = test_acc
                    if test_ex_zero_acc > gc.best.max_test_ex_zero_acc:
                        gc.best.max_test_ex_zero_acc = test_ex_zero_acc
                    if test_acc_5 > gc.best.max_test_acc_5:
                        gc.best.max_test_acc_5 = test_acc_5
                    if test_acc_7 > gc.best.max_test_acc_7:
                        gc.best.max_test_acc_7 = test_acc_7
                    if test_f1_mfn > gc.best.max_test_f1_mfn:
                        gc.best.max_test_f1_mfn = test_f1_mfn
                    if test_f1_raven > gc.best.max_test_f1_raven:
                        gc.best.max_test_f1_raven = test_f1_raven
                    if test_f1_muit > gc.best.max_test_f1_muit:
                        gc.best.max_test_f1_muit = test_f1_muit
            if best_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best': gc.best
                }, model_path)
            else:
                if epoch - gc.best.best_epoch > 10:
                    break

        tot_num = 0
        tot_err = 0
        tot_right = 0
        label_all = []
        output_all = []
        max_i = 0
        for i, data in enumerate(train_loader):
            batch_update_dict = {}
            max_i = i
            words, covarep, facet, inputLen, labels = data
            if covarep.size()[0] == 1:
                continue
            mask_ratio = np.random.choice([0, 0.2, 0.4], p=[0.7, 0.2, 0.1])
            if mask_ratio > 0:
                m = np.random.choice([0, 1], p=[mask_ratio, 1 - mask_ratio],
                                     size=(words.shape[0], words.shape[1], 1))
                words = words * torch.tensor(m.astype(np.float32))
            words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(
                device), inputLen.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, _ = net(words, covarep, facet)

            output_all.extend(outputs.tolist())
            label_all.extend(labels.tolist())
            if gc.dataset != "iemocap" or gc.dataset != "pom" or gc.dataset != "mosei_emo":
                err = torch.sum(torch.abs(outputs - labels))
                tot_right += torch.sum(torch.eq(torch.sign(labels), torch.sign(outputs)))
                tot_err += err
                tot_num += covarep.size()[0]
            if gc.dataset == 'iemocap':
                outputs = outputs.view(-1, 2)
                labels = labels.view(-1)
            loss = criterion(outputs, labels)
            # print("loss=%.4f, w_loss_item=%.4f, c_f_loss_item=%.4f" % (loss.item(), w_loss_item, c_f_loss_item))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=gc.config['max_grad'], norm_type=inf)
            if gc.save_grad and epoch in save_epochs:
                for name, param in net.named_parameters():
                    if param.grad is None:
                        continue
                    try:
                        if i == 0:
                            grad_dict[name] = param.grad.detach().cpu().numpy()
                        else:
                            grad_dict[name] = grad_dict[name] + np.abs(param.grad.detach().cpu().numpy())
                        assert (name not in batch_update_dict)
                        batch_update_dict[name] = param.data.detach().cpu().numpy()
                    except:
                        import pdb
                        pdb.set_trace()
            optimizer.step()
            if gc.save_grad and epoch in save_epochs:
                for name, param in net.named_parameters():
                    if param.grad is None:
                        continue
                    if i == 0:
                        update_dict[name] = np.abs(batch_update_dict[name] - param.data.detach().cpu().numpy())
                    else:
                        update_dict[name] += np.abs(batch_update_dict[name] - param.data.detach().cpu().numpy())
            running_loss += loss.item()
            del loss
            del outputs
            if i % 20 == 19:
                torch.cuda.empty_cache()
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        eval_method('train', output_all, label_all)

        if gc.save_grad and epoch in save_epochs:
            grad_f = h5py.File(os.path.join(gc.model_path, '%s_grad_%s_%d.hdf5' % (gc.dataset, config_name, epoch)))
            update_f = h5py.File(os.path.join(gc.model_path, '%s_update_%s_%d.hdf5' % (gc.dataset, config_name, epoch)))
            for name in grad_dict.keys():
                grad_avg = grad_dict[name] / (max_i + 1)
                grad_f.create_dataset(name, data=grad_avg)
                update_avg = update_dict[name] / (max_i + 1)
                update_f.create_dataset(name, data=update_avg)
            grad_f.close()
            update_f.close()


def eval_mosei_emo(split, output_all, label_all):
    truths = np.array(label_all).reshape((-1, len(gc.best.mosei_cls)))
    preds = np.array(output_all).reshape((-1, len(gc.best.mosei_cls)))
    cls_mae = {}
    for cls_id, cls in enumerate(gc.best.mosei_cls):
        mae = np.mean(np.absolute(preds[:, cls_id] - truths[:, cls_id]))
        cls_mae[cls] = mae
        print("\t%s %s mae: %f" % (split, cls, round(mae, 3)))
    return cls_mae


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


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
    print("\t%s correlation coefficient: %f" % (split, corr))
    print("\t%s accuracy: %f" % (split, acc))
    print("\t%s mult_acc_7: %f" % (split, acc_7))
    print("\t%s mult_acc_5: %f" % (split, acc_5))
    print("\t%s F1 MFN: %f " % (split, f1_mfn))
    print("\t%s F1 RAVEN: %f " % (split, f1_raven))
    print("\t%s F1 MuIT: %f " % (split, f1_muit))
    print("\t%s exclude zero accuracy: %f" % (split, ex_zero_acc))
    return mae, corr, acc, acc_7, acc_5, f1_mfn, f1_raven, f1_muit, ex_zero_acc


def logSummary():
    print("best epoch: %d" % gc.best.best_epoch)

    if gc.dataset == 'mosei_emo':
        for cls in gc.best.mosei_cls:
            print("best %s MAE: %f" % (cls, gc.best.mosei_emo_best_mae[cls]))

    else:
        print("best epoch: %d" % gc.best.best_epoch)
        print("lowest training MAE: %f" % gc.best.min_train_mae)
        print("lowest testing MAE: %f" % gc.best.min_test_mae)

        print("highest testing F1 MFN: %f" % gc.best.max_test_f1_mfn)
        print("highest testing F1 RAVEN: %f" % gc.best.max_test_f1_raven)
        print("highest testing F1 MuIT: %f" % gc.best.max_test_f1_muit)

        print("highest testing correlation: %f" % gc.best.max_test_cor)
        print("test correlation when validation correlation is the highest: %f" % gc.best.test_cor_at_valid_max)

        print("highest testing accuracy: %f" % gc.best.max_test_acc)

        print("highest testing exclude zero accuracy: %f" % gc.best.max_test_ex_zero_acc)

        print("highest testing accuracy 5: %f" % gc.best.max_test_acc_5)

        print("highest testing accuracy 7: %f" % gc.best.max_test_acc_7)



if __name__ == "__main__":
    start_time = time.time()
    print('Start time: ' + time.strftime("%H:%M:%S", time.gmtime(start_time)))

    parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
    parser.add_argument('-f', default='', type=str)
    parser.add_argument('--conf', type=str)
    parser.add_argument('--model-name', default=None, type=str)


    # Fixed
    parser.add_argument('--model', type=str, default='MulT',
                        help='name of the model to use (Transformer, etc.)')

    # Tasks
    parser.add_argument('--vonly', action='store_false',
                        help='use the crossmodal fusion into v (default: False)')
    parser.add_argument('--aonly', action='store_false',
                        help='use the crossmodal fusion into a (default: False)')
    parser.add_argument('--lonly', action='store_false',
                        help='use the crossmodal fusion into l (default: False)')
    parser.add_argument('--aligned', action='store_true', default=True,
                        help='consider aligned experiment or not (default: True)')

    # Dropouts
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                        help='attention dropout (for audio)')
    parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                        help='attention dropout (for visual)')
    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25,
                        help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.0,
                        help='output layer dropout')

    # Architecture
    parser.add_argument('--nlevels', type=int, default=5,
                        help='number of layers in the network (default: 5)')
    parser.add_argument('--num_heads', type=int, default=5,
                        help='number of heads for the transformer network (default: 5)')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')

    # Tuning
    parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                        help='batch size (default: 24)')
    parser.add_argument('--clip', type=float, default=0.8,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate (default: 1e-3)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=20,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--batch_chunk', type=int, default=1,
                        help='number of chunks per batch (default: 1)')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=30,
                        help='frequency of result logging (default: 30)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='do not use cuda')
    parser.add_argument('--name', type=str, default='mult',
                        help='name of the trial (default: "mult")')
    args = parser.parse_args()
    torch.manual_seed(gc.config['seed'])
    config_file_name = args.conf
    gc.config = json.load(open(config_file_name), object_pairs_hook=OrderedDict)
    model_name = args.model_name
    train_model(args, config_file_name, model_name)
    elapsed_time = time.time() - start_time
    print('Total time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))