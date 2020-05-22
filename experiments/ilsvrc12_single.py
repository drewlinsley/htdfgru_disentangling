import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 5  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'ilsvrc12',
    ]
    exp['val_dataset'] = [
        'ilsvrc12',
    ]
    exp['model'] = [
        # 'resnet_18',
        # 'resnet_50',
        # 'resnet_152',
        # 'unet',
        # 'seung_unet',
        # 'bu_fgru',
        # 'htd_fgru_linear',
        # 'td_fgru_linear',
        # 'h_fgru_linear',
        # 'htd_fgru',
        # 'h_fgru',
        # 'td_fgru',
        # 'htd_fgru_t1',
        # 'h_fgru_t1',
        # 'td_fgru_t1',
    ]
    exp['validation_period'] = [5000]
    exp['validation_steps'] = [3125]
    exp['shuffle_val'] = [False]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]

    # Model hyperparameters
    exp['loss_function'] = ['cce']
    exp['score_function'] = ['accuracy']
    exp['lr'] = [1e-5]
    exp['exclusion_lr'] = 1e-2
    exp['exclusion_scope'] = 'fgru'
    exp['optimizer'] = ['momentum']  # , 'adam']
    exp['optimizer'] = ['adam']  # , 'adam']
    # exp['lr_schedule'] = [{'ilsvrc12': [1281167, 128]}]
    # exp['variable_moving_average'] = 0.999
    # exp['lr_schedule'] = [{'superconvergence': [1e6, 16]}]
    exp['train_batch_size'] = [4] #  [32]
    exp['val_batch_size'] = [4] #  [32]
    exp['epochs'] = [100]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        # 'rgb2gray',
        # 'singleton',
        'random_crop',
        'left_right',
        # 'up_down',
        # 'pascal_normalize',
        'image_to_bgr',
        # 'uint8_rescale',
        # 'zero_one'
    ]]
    exp['val_augmentations'] = [[
        # 'rgb2gray',
        # 'singleton',
        'center_crop',
        'pascal_normalize',
        'image_to_bgr',
        # 'left_right',
        # 'up_down',
        # 'uint8_rescale',
        # 'zero_one'
    ]]
    return exp

