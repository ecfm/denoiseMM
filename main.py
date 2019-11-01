import json
import os
import signal
import sys
import time
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, f1_score

from consts import global_consts as gc
from model import Net
from adversary import Adv


lambda_q = 0.15
def stopTraining(signum, frame):
    global savedStdout
    logSummary()
    sys.stdout = savedStdout
    sys.exit()


def train_model(config_file_name, model_name):
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

    if gc.log_path != None:
        dir_path = "%s%d" % (gc.log_path, gc.HPID)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        log_file = "%s/print.log" % dir_path
        f = open(log_file, "w+")
        sys.stdout = f
    elif gc.dataset == 'mosei_emo':
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

    valid_dataset = ds(gc.data_path, cls="valid")
    valid_loader = Data.DataLoader(
        dataset=valid_dataset,
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

    net = Net()
    print(net)
    net.to(device)

    adv = Adv()

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
                outputs = net(words, covarep, facet, inputLen)

                w_output = adv(words)
                loss = crossentropy(w_output, labels) * lambda_q
                loss.backward()

                test_output_all.extend(outputs.tolist())
                test_label_all.extend(labels.tolist())
            if gc.dataset == 'mosei_emo':
                test_mae = eval_mosei_emo('test', test_output_all, test_label_all)
            else:
                test_mae, test_cor, test_acc, test_acc_7, test_acc_5, test_f1_mfn, test_f1_raven, test_f1_muit, \
                test_ex_zero_acc = eval_mosi('test', test_output_all, test_label_all)

            label_all = []
            output_all = []
            for data in valid_loader:
                words, covarep, facet, inputLen, labels = data
                if covarep.size()[0] == 1:
                    continue
                words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(
                    device), inputLen.to(device), labels.to(device)
                outputs = net(words, covarep, facet, inputLen)
                output_all.extend(outputs.data.cpu().tolist())
                label_all.extend(labels.data.cpu().tolist())
            best_model = False

            if gc.dataset == 'mosei_emo':
                for cls in gc.best.mosei_cls:
                    if test_mae[cls] < gc.best.mosei_emo_best_mae[cls]:
                        gc.best.mosei_emo_best_mae[cls] = test_mae[cls]
                        gc.best.best_epoch = epoch + 1
                        best_model = True
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
            words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(
                device), inputLen.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(words, covarep, facet, inputLen)
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

        if gc.dataset == 'mosei_emo':
            eval_mosei_emo('train', output_all, label_all)
        else:
            train_mae = tot_err / tot_num
            train_acc = float(tot_right) / tot_num
            print("\ttrain mean error: %f" % train_mae)
            print("\ttrain acc: %f" % train_acc)
            if train_mae < gc.best.min_train_mae:
                gc.best.min_train_mae = train_mae
            if train_acc > gc.best.max_train_acc:
                gc.best.max_train_acc = train_acc

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


def logSummary():
    print("best epoch: %d" % gc.best.best_epoch)

    if gc.dataset == 'mosei_emo':
        for cls in gc.best.mosei_cls:
            print("best %s MAE: %f" % (cls, gc.best.mosei_emo_best_mae[cls]))

    else:
        print("best epoch: %d" % gc.best.best_epoch)
        print("lowest training MAE: %f" % gc.best.min_train_mae)
        print("lowest testing MAE: %f" % gc.best.min_test_mae)
        print("lowest validation MAE: %f" % gc.best.min_valid_mae)
        print("test MAE when validation MAE is the lowest: %f" % gc.best.test_mae_at_valid_min)

        print("highest testing F1 MFN: %f" % gc.best.max_test_f1_mfn)
        print("highest testing F1 RAVEN: %f" % gc.best.max_test_f1_raven)
        print("highest testing F1 MuIT: %f" % gc.best.max_test_f1_muit)

        print("highest validation F1 MFN: %f" % gc.best.max_valid_f1_mfn)
        print("highest validation F1 RAVEN: %f" % gc.best.max_valid_f1_raven)
        print("highest validation F1 MuIT: %f" % gc.best.max_valid_f1_muit)

        print("test F1 MFN when validation F1 is the highest: %f" % gc.best.test_f1_mfn_at_valid_max)
        print("test F1 RAVEN when validation F1 is the highest: %f" % gc.best.test_f1_raven_at_valid_max)
        print("test F1 MuIT when validation F1 is the highest: %f" % gc.best.test_f1_muit_at_valid_max)

        print("highest testing correlation: %f" % gc.best.max_test_cor)
        print("highest validation correlation: %f" % gc.best.max_valid_cor)
        print("test correlation when validation correlation is the highest: %f" % gc.best.test_cor_at_valid_max)

        print("highest testing accuracy: %f" % gc.best.max_test_acc)
        print("highest validation accuracy: %f" % gc.best.max_valid_acc)
        print("test accuracy when validation accuracy is the highest: %f" % gc.best.test_acc_at_valid_max)

        print("highest testing exclude zero accuracy: %f" % gc.best.max_test_ex_zero_acc)
        print("highest validation exclude zero accuracy: %f" % gc.best.max_valid_ex_zero_acc)
        print("test ex-zero accuracy when validation ex-zero accuracy is the highest: %f" %
              gc.best.test_ex_zero_acc_at_valid_max)

        print("highest testing accuracy 5: %f" % gc.best.max_test_acc_5)
        print("highest validation accuracy 5: %f" % gc.best.max_valid_acc_5)
        print("test accuracy 5 when validation accuracy 5 is the highest: %f" % gc.best.test_acc_5_at_valid_max)

        print("highest testing accuracy 7: %f" % gc.best.max_test_acc_7)
        print("highest validation accuracy 7: %f" % gc.best.max_valid_acc_7)
        print("test accuracy 7 when validation accuracy 7 is the highest: %f" % gc.best.test_acc_7_at_valid_max)


if __name__ == "__main__":
    start_time = time.time()
    print('Start time: ' + time.strftime("%H:%M:%S", time.gmtime(start_time)))
    config_file_name = ''
    if len(sys.argv) > 1:
        config_file_name = sys.argv[1]
        gc.config = json.load(open(config_file_name), object_pairs_hook=OrderedDict)
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    else:
        model_name = None
    torch.manual_seed(gc.config['seed'])
    train_model(config_file_name, model_name)
    elapsed_time = time.time() - start_time
    print('Total time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))