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

    exp['validation_period'] = [1000]
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
    exp['train_loss_function'] = ['dummy']  # mirror_invariant_l2_grating']
    exp['val_loss_function'] = ['dummy']  # mirror_invariant_l2_grating']
    exp['score_function'] = ['dummy']  # mirror_invariant_l2_grating']
    exp['optimizer'] = ['adam_w']  # , 'adam']
    exp['early_stop'] = 1
    # exp['clip_gradients'] = 7
    exp['train_batch_size'] = [1]  # 10]
    exp['val_batch_size'] = [1]  # 10]
    exp['test_batch_size'] = [1]  # 10]
    exp['epochs'] = [1]
    exp['all_results'] = True
    # exp['plot_recurrence'] = True

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'singleton',
        # 'gratings_modulate',
        'stack3d',
        'resize',
        'pascal_normalize',
        # 'uint8_rescale',
        # 'zero_one',
    ]]
    exp['val_augmentations'] = [[
        'singleton',
        # 'gratings_modulate',
        'stack3d',
        'resize',
        'pascal_normalize',
        # 'uint8_rescale',
        # 'zero_one',
    ]]
    exp['test_augmentations'] = exp['val_augmentations']
    return exp
