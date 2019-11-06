import functools
import json
import random
import operator

random.seed(0)
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=False, type=int, default=0)
    ap.add_argument("--end", required=False, type=int, default=64)
    ap.add_argument("--gpus", required=False, type=int, default=2)
    args = vars(ap.parse_args())
    val_options = {
        "lr": [0.001],
        "seed": [0],
        "epoch_num": [200],
        "dropout": [0, 0.1],
        "n_head": [3, 4, 5], #, 8],
        "n_layers": [6, 7, 8]
    }
    option_lens = [len(options) for options in val_options.values()]
    num_comb = functools.reduce(operator.mul, option_lens, 1)
    start = int(args["start"])
    end = min(int(args["end"]), num_comb)
    gpus = int(args["gpus"])
    for i, id in enumerate(random.sample(range(num_comb), end)[start:end]):
        cuda = i % gpus
        file_name = 'configs/' + 'conf_cuda%d_%d.json' % (cuda, i + start)
        new_config = {'cuda': cuda}
        for key, options in val_options.items():
            choice = id % len(options)
            new_config[key] = options[choice]
            id //= len(options)
        with open(file_name, 'w') as outfile:
            json.dump(new_config, outfile)
