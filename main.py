import argparse
import json
import os
import signal
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from consts import global_consts as gc
from masked_dataset import MaskedDataset
from net import Net, set_requires_grad

lambda_q = 0.15
l_mode = True
av_mode = False

np.random.seed(0)

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8
}

def stopTraining():
    global savedStdout
    logSummary()
    sys.stdout = savedStdout
    sys.exit()

def get_test_metrics(epoch, device, test_loader, net):
    with torch.no_grad():
        print("Epoch #%d results:" % epoch)
        test_label_all = []
        test_output_l_all = []
        test_output_av_all = []
        test_output_all = []


        for data in test_loader:
            words, covarep, facet, inputLen, labels = data
            words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(device), \
                                                      inputLen.to(device), labels.to(device)
            if covarep.size()[0] == 1:
                continue
            if l_mode:
                outputs_l = net(x_l=words, train_l=True)
                test_output_l_all.extend(outputs_l.tolist())
            elif av_mode:
                outputs_av, outputs_l, _, _ = net(x_l=words, x_a=covarep, x_v=facet, train_av=True)
                test_output_av_all.extend(outputs_av.tolist())
                test_output_l_all.extend(outputs_l.tolist())
            else:
                outputs_av, outputs_l, outputs, _, _ = net(x_l=words, x_l_masked=words, x_a=covarep, x_v=facet)
                test_output_av_all.extend(outputs_av.tolist())
                test_output_l_all.extend(outputs_l.tolist())
                test_output_all.extend(outputs.tolist())
            test_label_all.extend(labels.tolist())

        best_model = False
        test_mae_av = 10
        test_mae_l = 10
        test_mae = 10
        if len(test_output_l_all) > 0:
            test_mae_l = eval_senti('test', 'l', test_output_l_all, test_label_all)
            if test_mae_l < gc.best.min_test_mae_l and l_mode:
                print("best mae l!!!!!!")
                gc.best.min_test_mae_l = test_mae_l
                if l_mode:
                    gc.best.best_epoch = epoch
                    best_model = True
        if len(test_output_av_all) > 0:
            # import pdb
            # pdb.set_trace()
            test_mae_av = eval_senti('test', 'av', test_output_av_all, test_label_all)
            if test_mae_av < gc.best.min_test_mae_av:
                print("best mae av!!!!!!")
                gc.best.min_test_mae_av = test_mae_av
                if av_mode:
                    gc.best.best_epoch = epoch
                    best_model = True
        if len(test_output_all) > 0:
            # import pdb
            # pdb.set_trace()
            test_mae = eval_senti('test', 'lav', test_output_all, test_label_all)
            if test_mae < gc.best.min_test_mae:
                gc.best.min_test_mae = test_mae
                gc.best.best_epoch = epoch
                best_model = True
        return best_model, test_mae_av, test_mae_l, test_mae


def train_model(args, config_file_name, model_name):
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
    # hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_dataset.__len__()
    net = Net()
    net.to(device)
    print(net)

    if gc.dataset == "iemocap":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=gc.config['lr'])
    start_epoch = 0
    model_path = os.path.join(gc.model_path, gc.dataset + '_' + model_name + '.tar')
    global l_mode, av_mode
    for epoch in range(start_epoch, 500):
        if epoch % 10 == 0:
            print("HPID:%d:Training Epoch %d." % (gc.HPID, epoch))
        if epoch % 100 == 0:
            logSummary()
        if gc.lr_decay and (epoch == 75 or epoch == 200):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2
        best_model, _, _, _ = get_test_metrics(epoch, device, test_loader, net)
        if best_model:
            torch.save({
                'epoch': epoch,
                'state': net.state_dict(),
                'best': gc.best
            }, model_path)
        else:
            if l_mode:
                if epoch - gc.best.best_epoch > 20:
                    print("!!!!!!!!!!!!!!!!STOP L-mode")
                    l_mode = False
                    av_mode = True
                    checkpoint = torch.load(model_path, map_location=device)
                    net.load_state_dict(checkpoint['state'])
                    get_test_metrics(epoch, device, test_loader, net)
                    # import pdb
                    # pdb.set_trace()
                    gc.best.best_epoch = epoch
                    set_requires_grad(net.proj_l, False)
                    set_requires_grad(net.enc_l, False)
                    set_requires_grad(net.dec_l, False)
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                           betas=(0.9, 0.98), eps=1e-09, lr=gc.config['lr'])
            elif av_mode:
                if epoch - gc.best.best_epoch > 60:
                    print("!!!!!!!!!!!!!!!!STOP AV-mode")
                    av_mode = False
                    checkpoint = torch.load(model_path, map_location=device)
                    net.load_state_dict(checkpoint['state'])
                    get_test_metrics(epoch, device, test_loader, net)
                    gc.best.best_epoch = epoch
                    set_requires_grad(net.proj_l, True)
                    set_requires_grad(net.enc_l, True)
                    set_requires_grad(net.dec_l, True)
                    optimizer = optim.Adam(net.parameters(),
                                           betas=(0.9, 0.98), eps=1e-09, lr=gc.config['lr'])
            else:
                if epoch - gc.best.best_epoch > 60:
                    break
        label_all = []
        output_l_all = []
        output_av_all = []
        output_all = []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            words, covarep, facet, masked_words, inputLen, labels = data
            words, covarep, facet, masked_words, inputLen, labels = words.to(device), covarep.to(device), facet.to(
                device), masked_words.to(device), inputLen.to(device), labels.to(device)
            if covarep.size()[0] == 1:
                continue
            if l_mode:
                outputs_l = net(x_l=words, train_l=True)
                loss_l = criterion(outputs_l, labels)
                loss_l.backward(retain_graph=True)
                output_l_all.extend(outputs_l.tolist())
            elif av_mode:
                outputs_av, _, l_latent, av2l_latent = net(x_l=words, x_a=covarep, x_v=facet, train_av=True)
                loss_av = criterion(l_latent, av2l_latent)
                loss_av.backward(retain_graph=True)
                output_av_all.extend(outputs_av.tolist())
            else:
                outputs_av, outputs_l, outputs, av2l_latent, l_latent = \
                    net(x_l=words, x_l_masked=masked_words, x_a=covarep, x_v=facet)
                loss_l = criterion(outputs_l, labels)
                loss_l.backward(retain_graph=True)
                output_l_all.extend(outputs_l.tolist())
                loss_av = criterion(l_latent, av2l_latent)
                loss_av.backward(retain_graph=True)
                output_av_all.extend(outputs_av.tolist())
                loss = criterion(outputs, labels)
                loss.backward(retain_graph=True)
                output_all.extend(outputs.tolist())
            # g = make_dot(outputs, dict(net.named_parameters()))
            # g.render('model/outputs_detach', view=True)

            optimizer.step()
            label_all.extend(labels.tolist())
            if i % 20 == 19:
                torch.cuda.empty_cache()
        if len(output_l_all) > 0:
            eval_senti('train', 'l', output_l_all, label_all)
        if len(output_av_all) > 0:
            eval_senti('train', 'av', output_av_all, label_all)
        if len(output_all) > 0:
            eval_senti('train', 'lav', output_all, label_all)
    # maes_av = []
    # maes_l = []
    # maes = []
    # checkpoint = torch.load(model_path, map_location=device)
    # net.load_state_dict(checkpoint['state'])
    # for mask_ratio in [0.2, 0.4, 0.6]:
    #     ds = MaskedDataset
    #     test_dataset = ds(gc.data_path, 'mosei_senti_%.0E_mask_data.pkl' % mask_ratio, cls="test")
    #     test_loader = Data.DataLoader(
    #         dataset=test_dataset,
    #         batch_size=100,
    #         shuffle=False,
    #         num_workers=1,
    #     )
    #     _, mae_av, mae_l, mae = get_test_metrics(-1, device, test_loader, net)
    #     maes_av.append(mae_av)
    #     maes_l.append(mae_l)
    #     maes.append(mae)
    # print("mask_ratio=[0, 0.2, 0.4, 0.6], maes:")
    # with open(os.path.join(gc.model_path, config_name + "_results.csv"), "w") as f:
    #     f.write("%s, l, %f,%f,%f,%f\n" % (config_name, gc.best.min_test_mae_l, maes_l[0], maes_l[1], maes_l[2]))
    #     f.write("%s, av, %f,%f,%f,%f\n" % (config_name, gc.best.min_test_mae_av, maes_av[0], maes_av[1], maes_av[2]))
    #     f.write("%s, lav, %f,%f,%f,%f\n" % (config_name, gc.best.min_test_mae, maes[0], maes[1], maes[2]))



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
    print("\t%s mae_%s : %f" % (split, mod, mae))
    return mae


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
    # try:
    train_model(args, config_file_name, model_name)
    # except Exception as e:
    #     print("Skip config {} because of '{}'".format(config_file_name, e))