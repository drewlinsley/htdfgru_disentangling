import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 5  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
    ]
    exp['val_dataset'] = [
    ]
    exp['model'] = [
    ]
    exp['validation_period'] = [0]
    exp['validation_steps'] = [520377]
    exp['shuffle_val'] = [False]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]

    # Model hyperparameters
    exp['lr'] = [0]
    exp['loss_function'] = ['mirror_invariant_l2_grating']
    exp['score_function'] = ['mirror_invariant_l2_grating']
    exp['optimizer'] = ['adam_w']  # , 'adam']
    exp['train_batch_size'] = [1]
    exp['val_batch_size'] = [1]
    exp['test_batch_size'] = [1]
    exp['epochs'] = [1]
    exp['all_results'] = True

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'sgl_label',
        'rc_image_label',
        'pascal_normalize',
        'image_to_bgr',
    ]]
    exp['val_augmentations'] = [[
        'sgl_label',
        'cc_image_label',
        'pascal_normalize',
        'image_to_bgr',
    ]]
    exp['test_augmentations'] = exp['val_augmentations']
    return exp

