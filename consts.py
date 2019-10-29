from metrics import Best


class global_consts():
    single_gpu = True
    load_model = False
    SDK_PATH = "../../../CMU-MultimodalSDK"

    save_grad = False

    dataset = "mosei_emo"
    data_path = None
    if dataset == "mosi_short":
        data_path = "../../MOSI/"
    else:
        data_path = "../data/"
    embed_path = "/Users/cask/Downloads/glove.840B.300d.txt"
    sentiment = "sad" # for IEMOCAP, choose from happy, angry, sad and neutral
    model_path = "../model/"

    log_path = None

    HPID = -1

    batch_size = 20

    padding_len = -1

    lr_decay = False

    # cellDim = 150
    # normDim = 100
    # hiddenDim = 300
    config = {
      "cuda": 1,
      "lr": 0.01,
      "epoch_num": 300,
      "dropout": 0.2,
      "seed": 0,
      "gru_lr": 0.001,
      "gru_dropout": 0.2,
      "max_grad": 0.1,
      "mod": "l",

      "n_head": 2,

      "proj_dim_a": 5,

      "proj_dim_v": 20,

      "n_layers": 2,


      "ff_dim_final": 16,

      "dim_total_proj": 8,

      "conv_dims": [3]
    }


    device = None

    best = Best()

    dim_l = -1
    dim_a = -1
    dim_v = -1

    def logParameters(self):
        print( "Hyperparameters:")
        for name in dir(global_consts):
            if name.find("__") == -1 and name.find("max") == -1 and name.find("min") == -1:
                print( "\t%s: %s" % (name, str(getattr(global_consts, name))))
