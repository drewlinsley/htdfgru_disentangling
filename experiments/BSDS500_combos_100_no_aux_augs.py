import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 1  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'BSDS500',
    ]
    exp['val_dataset'] = [
        'BSDS500',
    ]
    exp['model'] = [
    ]

    exp['validation_period'] = [20]
    exp['validation_steps'] = [200 / 1]
    exp['shuffle_val'] = [True]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]
    exp['get_map'] = [False]  # Get mean average precisions

    # Model hyperparameters
    exp['lr'] = [1e-6]
    exp['exclusion_lr'] = 3e-4
    exp['exclusion_scope'] = 'fgru'
    exp['train_loss_function'] = ['bi_bce']
    exp['val_loss_function'] = ['bi_bce']
    exp['score_function'] = ['pixel_error']
    exp['optimizer'] = ['adam']
    # exp['lr_schedule'] = [{'bsds': [1000, 2]}]
    exp['early_stop'] = 2000
    exp['train_batch_size'] = [1]
    exp['val_batch_size'] = [1]
    exp['epochs'] = [2048 * 2]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'hed_resize',
        'rc_image_label',
        'rotate90_image_label',
        'lr_flip_image_label',
        'ud_flip_image_label',
        'hed_brightness',
        'hed_contrast',
        'pascal_normalize',
        'image_to_bgr',
    ]]
    exp['val_augmentations'] = [[
        'cc_image_label',
        'pascal_normalize',
        'image_to_bgr',
    ]]
    return exp
