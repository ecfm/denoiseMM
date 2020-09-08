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

import gc
import pandas
import torch
import torch.utils.data as Data

GRID_ROOT = os.path.abspath(__file__ + '/../../../grid_search')

random.seed(0)

torch.backends.cudnn.benchmark = True


def eval(instance_dir, data_path):
    global gpus, grid_dir, devices, dataset_class
    with open(os.path.join(instance_dir, "params.json"), 'r') as f:
        config = json.load(f)
    model_path = os.path.join(instance_dir, "checkpoint.pytorch")
    if torch.cuda.is_available():
        if "gpus" in config.keys():
            gpus = config["gpus"]
        else:
            gpus = list(range(torch.cuda.device_count()))
        devices = ["cuda:{}".format(i) for i in gpus]
    else:
        devices = ["cpu"]
    model_module = importlib.import_module('models.%s.model' % config['model'])
    global model_class
    model_class = model_module.Model
    test_dataset = dataset_class(data_path, cls="test")
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=1,
    )
    checkpoint = torch.load(model_path)
    model = model_class(devices[0], dataset_class, **config["model_params"])
    model.load_state_dict(checkpoint['state'])
    model.to(devices[0])
    test_metrics, _ = model.get_results_no_grad(test_loader)
    print(test_metrics)


def get_dataset_class(config):
    if config['dataset'] in ['mosi', 'mosei']:
        dataset_module = 'multimodal_senti_dataset'
        module_path = "datasets.%s" % dataset_module
        return importlib.import_module(module_path).MultimodalSentiDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid search')
    parser.add_argument("config_path", metavar="CONFIG", type=str, help="config file path")
    parser.add_argument("data_path", metavar="DATA", type=str, help="data file path")

    args = parser.parse_args()
    eval(args.config_path, args.data_path)

