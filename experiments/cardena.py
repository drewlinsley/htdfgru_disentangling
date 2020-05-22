import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 5  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'cardena',
    ]
    exp['val_dataset'] = [
        'cardena',
    ]
    exp['model'] = [
        'cardena'
    ]
    exp['validation_period'] = [100]
    exp['validation_steps'] = [23]
    exp['shuffle_val'] = [False]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]

    # Model hyperparameters
    exp['lr'] = [1e-2]
    exp['loss_function'] = ['masked_mse']
    exp['score_function'] = ['masked_mse']
    exp['optimizer'] = ['adam']  # , 'adam']
    exp['train_batch_size'] = [32]
    exp['val_batch_size'] = [32]
    exp['test_batch_size'] = [32]
    exp['epochs'] = [200]
    exp['all_results'] = True

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'singleton',
    ]]
    exp['val_augmentations'] = [[
        'singleton',
    ]]
    exp['test_augmentations'] = exp['val_augmentations']
    return exp

