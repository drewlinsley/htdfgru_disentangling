import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 1  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'BSDS500_001',
        'BSDS500_010',
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
    exp['exclusion_lr'] = 1e-4
    # exp['train_loss_function'] = ['hed_bce']  # ['pearson']  # ['hed_bce']  # ['bsds_bce']
    # exp['val_loss_function'] = ['hed_bce']  # ['pearson']  # ['hed_bce']  # ['bsds_bce']  # ['bsds_bce']
    exp['train_loss_function'] = ['wasserstein']  # ['pearson']  # ['hed_bce']  # ['bsds_bce']
    exp['val_loss_function'] = ['wasserstein']  # ['pearson']  # ['hed_bce']  # ['bsds_bce']  # ['bsds_bce']
    # exp['train_loss_function'] = ['dice']  # ['hed_bce']  # ['bsds_bce']
    # exp['val_loss_function'] = ['dice']  # ['hed_bce']  # ['bsds_bce']  # ['bsds_bce']
    exp['default_restore'] = True
    exp['score_function'] = ['pixel_error']  # ['bsds_f1']
    # exp['optimizer'] = ['adam_w']  # ['momentum']
    exp['optimizer'] = ['adam']  # ['momentum']
    exp['lr_schedule'] = [{'bsds': [200, 1]}]
    exp['optimizer'] = ['momentum']  # , 'adam']
    # exp['optimizer'] = ['adam']  # , 'adam']
    exp['lr_schedule'] = [{'ilsvrc12': [200, 2]}]
    exp['exclusion_scope'] = 'contour_readout'
    exp['early_stop'] = 50
    # exp['clip_gradients'] = 7
    exp['train_batch_size'] = [2]  # 10]
    exp['val_batch_size'] = [2]  # 10]
    exp['epochs'] = [2048 * 2]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'lr_flip_image_label',
        # 'rot_image_label',
        'ilsvrc12_normalize',
        # 'bsds_normalize',
        'res_nn_image_label',
        # 'blur_labels',
        # 'uint8_rescale',
        # 'zero_one',
        # 'bfloat16',
    ]]
    exp['val_augmentations'] = [[
        # 'bsds_mean',
        # 'bsds_normalize',
        'ilsvrc12_normalize',
        'res_nn_image_label',
        # 'res_image_label',
        # 'blur_labels',
        # 'uint8_rescale',
        # 'zero_one',
    ]]
    return exp

