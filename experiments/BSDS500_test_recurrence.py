import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 1  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'BSDS500_test_landscape',
    ]
    exp['val_dataset'] = [
        'BSDS500_test_landscape',
    ]
    exp['model'] = [
        'refactored_v4'
        # 'hgru_bn_bsds'
        # 'fgru_bsds'
    ]

    exp['validation_period'] = [100]
    exp['validation_steps'] = [200 / 8]
    exp['shuffle_val'] = [False]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]

    # Model hyperparameters
    exp['lr'] = [0.]
    # exp['exclusion_lr'] = 1e-4
    exp['train_loss_function'] = ['hed_bce']  # ['pearson']  # ['hed_bce']  # ['bsds_bce']
    exp['val_loss_function'] = ['hed_bce']  # ['pearson']  # ['hed_bce']  # ['bsds_bce']  # ['bsds_bce']
    exp['score_function'] = ['pixel_error']  # ['bsds_f1']
    exp['optimizer'] = ['adam']  # ['momentum']
    exp['early_stop'] = 1
    # exp['clip_gradients'] = 7
    exp['train_batch_size'] = [1]  # 10]
    exp['val_batch_size'] = [1]  # 10]
    exp['test_batch_size'] = [1]  # 10]
    exp['epochs'] = [1]
    exp['all_results'] = True
    exp['plot_recurrence'] = True

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        # 'rot_image_label',
        # 'ilsvrc12_normalize',
        'sgl_label',
        'image_to_bgr',
        'pascal_normalize',
        'cc_image_label',
        # 'bsds_normalize',
        # 'res_nn_image_label',
        # 'blur_labels',
        # 'uint8_rescale',
        # 'zero_one',
        # 'bfloat16',
    ]]
    exp['val_augmentations'] = [[
        # 'bsds_mean',
        # 'bsds_normalize',
        # 'ilsvrc12_normalize',
        'sgl_label',
        'image_to_bgr',
        'pascal_normalize',
        'cc_image_label',
        # 'res_nn_image_label',
        # 'res_image_label',
        # 'blur_labels',
        # 'uint8_rescale',
        # 'zero_one',
    ]]
    exp['test_augmentations'] = exp['val_augmentations']
    return exp
