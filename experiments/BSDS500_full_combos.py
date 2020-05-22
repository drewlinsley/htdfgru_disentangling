import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 1  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'BSDS500_100',
    ]
    exp['val_dataset'] = [
        'BSDS500_100',
    ]
    exp['model'] = [
        'refactored_v4'
        # 'hgru_bn_bsds'
        # 'fgru_bsds'
    ]

    exp['validation_period'] = [50]
    exp['validation_steps'] = [20]
    exp['shuffle_val'] = [False]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]
    exp['get_map'] = [True]  # Get mean average precisions

    # Model hyperparameters
    exp['lr'] = [1e-2]
    exp['train_loss_function'] = ['dice']  # ['bsds_bce']  # ['hed_bce']  # ['bsds_bce']
    exp['val_loss_function'] = ['dice']  # ['bsds_bce']  # ['bsds_bce']
    exp['score_function'] = ['pixel_error']  # ['bsds_f1']
    # exp['optimizer'] = ['adam_w']  # ['momentum']
    exp['optimizer'] = ['adam']  # ['momentum']
    exp['lr_schedule'] = [{'bsds': [2, 1]}]
    exp['exclusion_scope'] = 'contour_readout'
    exp['early_stop'] = 30
    exp['train_batch_size'] = [1]
    exp['val_batch_size'] = [1]
    exp['epochs'] = [2048 * 2]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'ud_flip_image_label',
        'lr_flip_image_label',
        # 'rot_image_label',
        'ilsvrc12_normalize',
        # 'bsds_normalize',
        'random_scale_crop_image_label',
        # 'rc_image_label',
        'blur_labels',
        'ilsvrc12_normalize',
        # 'uint8_rescale',
        # 'zero_one',
        # 'bfloat16',
    ]]
    exp['val_augmentations'] = [[
        # 'bsds_mean',
        # 'bsds_normalize',
        'ilsvrc12_normalize',
        'cc_image_label',
        'blur_labels',
        'ilsvrc12_normalize',
        # 'uint8_rescale',
        # 'zero_one',
    ]]
    return exp
