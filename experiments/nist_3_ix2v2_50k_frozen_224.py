import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 3  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'cluttered_nist_3_ix2v2_50k_224',
    ]
    exp['val_dataset'] = [
        'cluttered_nist_3_ix2v2_50k_224',
    ]
    exp['model'] = [
        # 'h_fgru_linear',
        # 'htd_fgru',
        'h_fgru',
        'td_fgru',
        # 'htd_fgru_t1',
        # 'h_fgru_t1',
        # 'td_fgru_t1',
    ]
    exp['validation_period'] = [2000]
    exp['validation_steps'] = [625]
    exp['shuffle_val'] = [True]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [True]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]

    # Model hyperparameters
    exp['lr'] = [3e-4]
    # exp['lr'] = [1e-3, 1e-4, 1e-5, 1e-6]
    exp['loss_function'] = ['bce']
    exp['val_loss_function'] = exp['loss_function']
    exp['score_function'] = ['accuracy']
    exp['optimizer'] = ['nadam']  # , 'adam']
    exp['train_batch_size'] = [32]
    exp['val_batch_size'] = [32]
    exp['epochs'] = [100]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        # 'grayscale',
        'singleton',
        'stack3d',
        'resize',
        'pascal_normalize',
        'image_to_bgr',
        # 'left_right',
        # 'up_down',
        # 'singleton',
        # 'zero_one'
    ]]
    exp['val_augmentations'] = [[
        'singleton',
        'stack3d',
        'resize',
        'pascal_normalize',
        'image_to_bgr',
    ]]
    return exp
