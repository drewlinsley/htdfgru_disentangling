import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 10  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'new_LMD_512',
    ]
    exp['val_dataset'] = [
        'new_LMD_512',
    ]
    exp['model'] = [
        # 'resnet_18',
        'htd_fgru_v2_fancy'
        # 'hgru_slim_learned'
        # 'hgru_slim_color'
    ]
    bs = 4  # 32
    exp['validation_period'] = [50]
    exp['validation_steps'] = [130 / bs]
    exp['shuffle_val'] = [False]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]
    exp['get_map'] = [False]  # Get mean average precisions

    # Model hyperparameters
    exp['loss_function'] = ['bce']
    exp['score_function'] = ['accuracy']
    exp['lr'] = [1e-3]
    # exp['optimizer'] = ['momentum']  # , 'adam']
    exp['default_restore'] = True
    # exp['lr'] = [3e-4]
    exp['optimizer'] = ['nadam']  # , 'adam']
    exp['train_batch_size'] = [bs]
    exp['val_batch_size'] = [bs]
    exp['test_batch_size'] = [bs]
    exp['epochs'] = [1000]
    exp['get_lr_schedule'] = ['ilsvrc12']

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        # 'singleton',
        # 'uint8_rescale',
        'rgb2gray',
        'rotate90',
        'left_right',
        'up_down',
        'random_crop',
        # 'zero_one',
    ]]
    exp['val_augmentations'] = [[
        # 'singleton',
        # 'uint8_rescale',
        'rgb2gray',
        'center_crop',
        # 'zero_one',
    ]]
    exp['test_augmentations'] = exp['val_augmentations']
    return exp
