import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 1  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'BSDS500_001',
    ]
    exp['val_dataset'] = [
        'BSDS500_010',
    ]
    exp['model'] = [
        'refactored_v1'
        # 'hgru_bn_bsds'
        # 'fgru_bsds'
    ]

    exp['validation_period'] = [500]
    exp['validation_steps'] = [20]
    exp['shuffle_val'] = [True]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]

    # Model hyperparameters
    exp['lr'] = [1e-2]
    exp['loss_function'] = ['dice']  # ['bsds_bce']
    exp['score_function'] = ['sigmoid_pearson']  # ['bsds_f1']
    exp['optimizer'] = ['adam_w']  # ['momentum']
    # exp['lr_schedule'] = [{'bsds': [200, 2]}]
    exp['train_batch_size'] = [2]
    exp['val_batch_size'] = [2]
    exp['epochs'] = [2048 * 2]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        # 'ud_flip_image_label',
        # 'rot_image_label',
        'bsds_normalize',
        'rc_image_label',
        'blur_labels',
        # 'uint8_rescale',
        # 'zero_one',
        # 'bfloat16',
    ]]
    exp['val_augmentations'] = [[
        # 'bsds_mean',
        'bsds_normalize',
        'cc_image_label',
        'blur_labels',
        # 'uint8_rescale',
        # 'zero_one',
    ]]
    return exp
