import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 5  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'cube_plus',
    ]
    exp['val_dataset'] = [
        'cube_plus',
    ]
    exp['model'] = [
        'hgru_color',
    ]

    exp['validation_period'] = [500]
    exp['validation_steps'] = [170 // 2]
    exp['shuffle_val'] = [True]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]

    # Model hyperparameters
    exp['lr'] = [1e-2]
    exp['loss_function'] = ['mse']
    exp['score_function'] = ['mse']
    exp['optimizer'] = ['nadam']
    exp['train_batch_size'] = [2]
    exp['val_batch_size'] = [2]
    exp['epochs'] = [12]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        # 'grayscale',
        'left_right',
        # 'up_down',
        # 'rotate90',
        'cube_plus_rescale',
        'random_crop_and_res_cube_plus',  # 'random_crop',
    ]]
    exp['val_augmentations'] = [[
        'cube_plus_rescale',
        'center_crop_and_res_cube_plus',  # 'center_crop'
    ]]
    return exp
