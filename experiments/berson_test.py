import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 1  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'berson_001',
        'berson_010',
        'berson_100',
    ]
    exp['val_dataset'] = [
        'berson_100'
    ]
    exp['model'] = [
        'seung_unet_per_pixel_instance',
        'refactored_v1'
    ]
    exp['validation_period'] = [50]
    exp['validation_steps'] = [191]
    exp['shuffle_val'] = [False]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]
    exp['get_map'] = [True]  # Get mean average precisions

    # Model hyperparameters
    exp['lr'] = [0]
    # exp['train_loss_function'] = ['berson_bce']  # ['sparse_cce_image']
    # exp['val_loss_function'] = ['connectomics_bce']  # ['sparse_cce_image']
    # exp['score_function'] = ['snemi_f1']
    exp['train_loss_function'] = ['l2']
    exp['val_loss_function'] = ['l2']  # ['sparse_cce_image']
    exp['test_loss_function'] = ['l2']  # ['sparse_cce_image']
    exp['score_function'] = ['mse']
    exp['optimizer'] = ['sgd']  # , 'adam']
    # exp['optimizer'] = ['momentum']  # , 'adam']
    # exp['optimizer'] = ['adam', 'adam_w']  # , 'adam']
    exp['train_batch_size'] = [1]
    exp['val_batch_size'] = [1]
    exp['test_batch_size'] = [1]
    exp['epochs'] = [1000]
    exp['early_stop'] = 10
    # exp['variable_moving_average'] = 0.999
    # exp['lr_schedule'] = {
    #     'connectomics_learning_rate_schedule': [3, exp['train_batch_size'][0]]}
    # exp['inclusion_scope'] = 't0'
    exp['plot_recurrence'] = True

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'singleton',
        # 'sgl_label',
        # 'res_image_label',
        # 'left_right',
        # 'up_down',
        'uint8_rescale',
        'zero_one',
        # 'threshold_label'
    ]]
    exp['val_augmentations'] = [[
        'singleton',
        # 'sgl_label',
        # 'res_image_label',
        # 'left_right',
        # 'up_down',
        'uint8_rescale',
        'zero_one',
        # 'threshold_label'
    ]]
    exp['test_augmentations'] = exp['val_augmentations']
    return exp
