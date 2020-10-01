import argparse
import hashlib
import importlib
import itertools
import json
import os
import random
import threading
import time
import traceback
from queue import Queue

import numpy as np
import gc
import pandas
import torch
import torch.utils.data as Data

GRID_ROOT = os.path.abspath(__file__ + '/../../../grid_search')

random.seed(0)

torch.backends.cudnn.benchmark = True


def eval(instance_dir, data_path, zero_mods):
    with open(os.path.join(os.path.dirname(instance_dir), "config.json"), 'r') as f:
        config = json.load(f)
    with open(os.path.join(instance_dir, "params.json"), 'r') as f:
        params = json.load(f)
    model_path = os.path.join(instance_dir, "checkpoint.pytorch")
    if torch.cuda.is_available():
        if "gpus" in config.keys():
            gpus = config["gpus"]
        else:
            gpus = list(range(torch.cuda.device_count()))
        devices = ["cuda:{}".format(i) for i in gpus]
    else:
        devices = ["cpu"]
    model_module = importlib.import_module('models.%s.model' % 'my_model_av2l_loss_simplified')
    model_class = model_module.Model
    dataset_class = get_dataset_class(config)
    test_dataset = dataset_class(data_path, zero_mods, cls="test")
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=1,
    )
    checkpoint = torch.load(model_path)
    model = model_class(devices[0], dataset_class, **params["model_params"])
    model.load_state_dict(checkpoint['state'])
    model.to(devices[0])
    print(f"epoch {checkpoint['epoch']}")
    test_metrics, test_output = model.get_results_no_grad(test_loader)
    print(f"saved test {checkpoint['test_metrics']}")
    print(checkpoint['test_outputs'][:100])
    print(f"test {test_metrics}")
    print(test_output[:100])


    valid_dataset = dataset_class(data_path, cls="valid")
    valid_loader = Data.DataLoader(
        dataset=valid_dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=1,
    )
    valid_metrics, valid_output = model.get_results_no_grad(valid_loader)
    print(f"saved valid {checkpoint['valid_metrics']}")
    print(checkpoint['valid_outputs'][:100])
    print(f"valid {valid_metrics}")
    print(valid_output[:100])


def get_dataset_class(config):
    if config['dataset'] in ['mosi', 'mosei']:
        dataset_module = 'multimodal_zero_mod_dataset'
        module_path = "datasets.%s" % dataset_module
        return importlib.import_module(module_path).MultimodalSentiDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid search')
    parser.add_argument("config_path", metavar="CONFIG", type=str, help="config file path")
    parser.add_argument("data_path", metavar="DATA", type=str, help="data file path")
    parser.add_argument("zero_mods", metavar="ZEROS", type=str, help="modalities to be masked")

    args = parser.parse_args()
    eval(args.config_path, args.data_path, args.zero_mods.split(','))

