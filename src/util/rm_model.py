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

GRID_ROOT = os.path.abspath(__file__ + '../../../../grid_search')
print(GRID_ROOT)
for conf_dir in os.scandir(GRID_ROOT):
    all_results_path = os.path.join(conf_dir, "all_results.csv")
    if not os.path.exists(all_results_path):
        continue
    print(f"Removing models in {conf_dir.path}")
    all_results = pandas.read_csv(all_results_path)
    all_results = all_results.sort_values('test.mae')
    for id in all_results['id'].to_list()[5:]:
        model_path = os.path.join(conf_dir, id, "checkpoint.pytorch")
        if os.path.exists(model_path):
            os.remove(model_path)
