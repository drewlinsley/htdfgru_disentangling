import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 1  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'ilsvrc12',
    ]
    exp['val_dataset'] = [
        'ilsvrc12',
    ]
    exp['model'] = [
    ]
    exp['validation_period'] = [1000]
    exp['validation_steps'] = [780]
    exp['shuffle_val'] = [False]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [True]
    exp['save_gradients'] = [False]

    # Model hyperparameters
    exp['lr'] = [3e-4]
    exp['loss_function'] = ['contrastive']
    exp['score_function'] = ['contrastive']  # ['accuracy']
    exp['optimizer'] = ['momentum']
    # exp['lr_schedule'] = [{'ilsvrc12': [1281167, 128]}]
    exp['train_batch_size'] = [16]
    exp['val_batch_size'] = [16]
    exp['epochs'] = [100]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        # 'rgb2gray',
        # 'singleton',
        'left_right',
        # 'up_down',
        # 'rotate90',
        'uint8_rescale',
        'random_crop',
        # 'zero_one',
        'contrastive_loss'
    ]]
    exp['val_augmentations'] = [[
        # 'rgb2gray',
        # 'left_right',
        # 'up_down',
        # 'singleton',
        'uint8_rescale',
        'center_crop',
        # 'zero_one',
        'contrastive_loss'
    ]]
    return exp
