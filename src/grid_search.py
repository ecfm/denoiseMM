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

GRID_ROOT = os.path.abspath(__file__ + '/../../grid_search')

random.seed(0)

torch.backends.cudnn.benchmark = True

done_ids = set()
lock = threading.Lock()


# TODO: multithread result file conflict
def training_thread(device_idx, config):
    global all_results
    while True:
        device = devices[device_idx]
        model_params, train_params = queue.get()
        params = {"model_params": model_params, "train_params": train_params}
        instance_id = hashlib.md5(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()
        try:
            with lock:
                instance_dir = os.path.join(grid_dir, instance_id)
                result_file = os.path.join(instance_dir, "result.json")
                skip_file = os.path.join(instance_dir, "skip.json")
                if os.path.exists(result_file) or os.path.exists(skip_file):
                    print("Configuration {} has already been run, skip...".format(instance_id))
                    queue.task_done()
                    continue
                os.makedirs(instance_dir, exist_ok=True)
                with open(os.path.join(instance_dir, 'params.json'), 'w') as f:
                    json.dump(params, f, sort_keys=True)
            print("Beginning training instance on {}... with id {} model_params={}, train_params={}".format(
                device, instance_id, model_params, train_params))
            print("Save grid search results to {}".format(os.path.abspath(instance_dir)))
            start = time.perf_counter()
            train_dataset = dataset_class(config['data_path'], cls="train")
            train_loader = Data.DataLoader(
                dataset=train_dataset,
                batch_size=train_params['batch_size'],
                shuffle=True,
                num_workers=1,
            )

            valid_dataset = dataset_class(config['data_path'], cls="valid")
            valid_loader = Data.DataLoader(
                dataset=valid_dataset,
                batch_size=train_params['batch_size'],
                shuffle=False,
                num_workers=1,
            )

            test_dataset = dataset_class(config['data_path'], cls="test")
            test_loader = Data.DataLoader(
                dataset=test_dataset,
                batch_size=train_params['batch_size'],
                shuffle=False,
                num_workers=1,
            )
            model = model_class(device, dataset_class, **model_params)
            model.to(device)
            del train_params['batch_size']
            logs, best_result = model.train_eval(instance_dir, train_loader, valid_loader, test_loader, **train_params)
            end = time.perf_counter()
            run_time = end - start
            del model
        except Exception as e:
            if type(e) == ValueError and str(e) == "SKIP":
                print("Skip current invalid configuration on {} with "
                      "model_params={}, train_params={} because of {}".format(device, model_params, train_params, e))
                with open(skip_file, 'w') as f:
                    json.dump({'error': str(e)}, f, sort_keys=True)
                continue
            tb = traceback.format_exc()
            print(tb)
            print("WARNING: exception raised while training on {} with "
                  "model_params={}, train_params={}".format(device, model_params, train_params))
            with open(os.path.join(instance_dir, 'ERROR.json'), 'w') as f:
                json.dump({'error': str(e), 'traceback': tb}, f, sort_keys=True)
        else:
            pandas.DataFrame(logs).to_csv(os.path.join(instance_dir, "metrics.csv"), index=False)
            new_result = {"id": instance_id, "run_time": run_time,
                          **{"model." + k: v for k, v in model_params.items()},
                          **{"train." + k: v for k, v in train_params.items()},
                          **best_result}
            with open(result_file, 'w') as f:
                json.dump(new_result, f, sort_keys=True)
            with lock:
                all_results = all_results.append(new_result, ignore_index=True)
                all_results.to_csv(all_results_path, index=False)
            print("Training instance complete with result:", new_result)
        torch.cuda.empty_cache()
        gc.collect()
        queue.task_done()


def grid_search(config_path):
    global gpus, grid_dir, devices, dataset_class
    with open(config_path, "r") as f:
        config = json.load(f)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    grid_dir = os.path.join(GRID_ROOT, config_name)
    os.makedirs(grid_dir, exist_ok=True)
    with open(os.path.join(grid_dir, 'config.json'), 'w') as f:
        json.dump(config, f, sort_keys=True)

    dataset_class = get_dataset_class(config)
    max_threads = config["max_threads"]
    if torch.cuda.is_available():
        if "gpus" in config.keys():
            gpus = config["gpus"]
        else:
            gpus = list(range(torch.cuda.device_count()))
        devices = ["cuda:{}".format(i) for i in gpus]
    else:
        devices = ["cpu"]
    # get grid search combinations
    model_params = config["model_params"]
    train_params = config["train_params"]
    model_params_len = len(model_params.keys())
    # generate every possible combinations of all possible dataset_params and model_params
    combos = [(
               dict(zip(model_params.keys(), values[:model_params_len])),
               dict(zip(train_params.keys(), values[model_params_len:])))
              for values in itertools.product(*model_params.values(), *train_params.values())]
    random.shuffle(combos)
    global queue
    queue = Queue()
    for combo in combos:
        queue.put(combo)

    global all_results_path, all_results
    all_results_path = os.path.join(grid_dir, "all_results.csv")
    if os.path.isfile(all_results_path):
        all_results = pandas.read_csv(all_results_path)
    else:
        all_results = pandas.DataFrame()

    model_module = importlib.import_module('models.%s.model' % config['model'])
    global model_class
    model_class = model_module.Model

    print("Start grid search with %d combos" % len(combos))
    for i in range(max_threads):
        thread = threading.Thread(target=training_thread, args=(i % len(devices), config))
        thread.setDaemon(True)
        thread.start()
    queue.join()
    print("Grid search complete!")


def get_dataset_class(config):
    if config['dataset'] in ['mosi', 'mosei']:
        dataset_module = 'multimodal_senti_dataset'
        module_path = "datasets.%s" % dataset_module
        return importlib.import_module(module_path).MultimodalSentiDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid search')
    parser.add_argument("config_path", metavar="CONFIG", type=str, help="config file path")
    args = parser.parse_args()
    grid_search(args.config_path)

