import json
import os
from collections import OrderedDict
import csv

def get_metrics(fname):
    min_mae = 10
    with open(fname) as f:
        for line in f.readlines():
            line_start = '\ttest mean error: '
            if line.startswith(line_start):
                mae = float(line.split(line_start)[1])
                if mae < min_mae:
                    min_mae = mae
    return min_mae


LOG_DIR = 'logs/'
CONF_DIR = 'configs/'
count = 0
lines = []
headers = []
for log_file in os.listdir(LOG_DIR):
    config_name = os.path.splitext(os.path.basename(log_file))[0]
    if not config_name.startswith('conf'):
        continue
    config_file = CONF_DIR + config_name + '.json'
    if not os.path.exists(config_file):
        continue
    mae = get_metrics(LOG_DIR + log_file)

    config = json.load(open(config_file), object_pairs_hook=OrderedDict)
    if count == 0:
        headers = list(config.keys())
        lines.append(['config'] + headers + ['MAE'])
        count += 1
    lines.append([config_file] + list(config.values()) + [mae])

for l in lines:
    print(l)

with open('./result.csv', 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(lines)
