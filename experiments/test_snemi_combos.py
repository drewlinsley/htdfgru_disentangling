import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 10  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'snemi_200_test',
    ]
    exp['val_dataset'] = [
        'snemi_200_test',
    ]
    exp['model'] = [
        'seung_unet_per_pixel',
        'gammanet_t8_per_pixel',
    ]
    exp['validation_period'] = [25]
    exp['validation_steps'] = [40]
    exp['shuffle_val'] = [False]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]
    exp['get_map'] = [True]  # Get mean average precisions

    # Model hyperparameters
    exp['lr'] = [1e-3]
    exp['loss_function'] = ['sparse_cce_image']
    exp['score_function'] = ['f1']
    exp['optimizer'] = ['nadam']  # , 'adam']
    exp['train_batch_size'] = [16]
    exp['val_batch_size'] = [16]
    exp['test_batch_size'] = [16]
    exp['epochs'] = [100]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'singleton',
        'sgl_label',
        # 'res_image_label',
        # 'left_right',
        # 'up_down',
        'uint8_rescale',
        'zero_one',
        'threshold_label'
    ]]
    exp['val_augmentations'] = [[
        'singleton',
        'sgl_label',
        # 'res_image_label',
        # 'left_right',
        # 'up_down',
        'uint8_rescale',
        'zero_one',
        'threshold_label'
    ]]
    exp['test_augmentations'] = exp['val_augmentations']
    return exp
