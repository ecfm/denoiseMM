class Best():
    best_epoch = 0

    max_train_f1 = 0
    max_test_f1 = 0
    max_valid_f1 = 0

    max_valid_f1_mfn = 0
    max_valid_f1_raven = 0
    max_valid_f1_muit = 0

    max_test_f1_mfn = 0
    max_test_f1_raven = 0
    max_test_f1_muit = 0

    max_test_prec = 0
    max_valid_prec = 0
    max_train_prec = 0
    max_train_recall = 0
    max_test_recall = 0
    max_valid_recall = 0
    max_train_acc = 0
    max_valid_acc = 0
    max_test_acc = 0
    max_valid_ex_zero_acc = 0
    max_test_ex_zero_acc = 0
    max_valid_acc_5 = 0
    max_test_acc_5 = 0
    max_valid_acc_7 = 0
    max_test_acc_7 = 0

    test_acc_at_valid_max = 0
    test_ex_zero_acc_at_valid_max = 0
    test_acc_5_at_valid_max = 0
    test_acc_7_at_valid_max = 0
    test_f1_at_valid_max = 0

    test_f1_mfn_at_valid_max = 0
    test_f1_raven_at_valid_max = 0
    test_f1_muit_at_valid_max = 0

    test_prec_at_valid_max = 0
    test_recall_at_valid_max = 0

    min_train_mae = 10
    min_test_mae = 10
    min_test_mae_av = 10
    min_test_mae_l = 10
    max_test_cor = 0
    min_valid_mae = 10
    max_valid_cor = 0
    test_mae_at_valid_min = 10
    test_cor_at_valid_max = 0

    mosei_cls = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

    mosei_emo_best_mae = {cls: 10 for cls in ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']}

