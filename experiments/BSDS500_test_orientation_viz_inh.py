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

    exp['validation_period'] = [50]
    exp['validation_steps'] = [1]
    exp['shuffle_val'] = [False]  # Shuffle val data.
    exp['shuffle_train'] = [False]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]

    # Model hyperparameters
    exp['lr'] = [1]  # [10000]  # [100]
    # exp['lr'] = [10]  # [100]
    exp['stack_label_image'] = True
    # exp['exclusion_lr'] = 1e-4
    exp['train_loss_function'] = ['l2_viz_phase']  # ['hed_bce']  # ['pearson']  # ['hed_bce']  # ['bsds_bce']
    exp['val_loss_function'] = ['l2_viz_phase']  # ['hed_bce']  # ['pearson']  # ['hed_bce']  # ['bsds_bce']  # ['bsds_bce']
    exp['score_function'] = ['pass']  # ['bsds_f1']
    exp['optimizer'] = ['adam']
    exp['early_stop'] = 1000000
    # exp['clip_gradients'] = 7
    exp['train_batch_size'] = [1]  # 10]
    exp['val_batch_size'] = [1]  # 10]
    exp['test_batch_size'] = [1]  # 10]
    exp['epochs'] = [751 * 4]  # 40000
    exp['all_results'] = True
    # exp['plot_recurrence'] = True

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        # 'singleton',
        # 'gratings_modulate',
        # 'stack3d',
        # 'resize',
        # 'pascal_normalize',
        # 'lr_viz_flip',
    ]]
    exp['val_augmentations'] = [[
        # 'singleton',
        # 'gratings_modulate',
        # 'stack3d',
        # 'resize',
        # 'pascal_normalize',
        # 'lr_viz_flip'
    ]]
    exp['test_augmentations'] = exp['val_augmentations']
    return exp
