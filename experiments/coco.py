import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 1  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'coco',
    ]
    exp['val_dataset'] = [
        'coco',
    ]
    exp['model'] = [
        'refactored_v6'
    ]

    exp['validation_period'] = [500]
    exp['validation_steps'] = [100]
    exp['shuffle_val'] = [True]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]
    exp['force_path'] = True
    exp['get_map'] = [False]  # Get mean average precisions

    # Model hyperparameters
    exp['lr'] = [3e-4]
    exp['exclusion_lr'] = 3e-4
    exp['train_loss_function'] = ['sparse_ce_image']  # ['pearson']  # ['hed_bce']  # ['bsds_bce']
    exp['val_loss_function'] = ['sparse_ce_image']  # ['pearson']  # ['hed_bce']  # ['bsds_bce']  # ['bsds_bce']
    # exp['train_loss_function'] = ['dice']  # ['hed_bce']  # ['bsds_bce']
    # exp['val_loss_function'] = ['dice']  # ['hed_bce']  # ['bsds_bce']  # ['bsds_bce']
    exp['score_function'] = ['accuracy']  # ['bsds_f1']
    exp['optimizer'] = ['adam']  # ['momentum']
    # exp['optimizer'] = ['momentum']
    exp['default_restore'] = True
    # exp['lr_schedule'] = [{'bsds': [50000, 1]}]
    exp['exclusion_scope'] = 'contour_readout'
    exp['early_stop'] = 300
    # exp['clip_gradients'] = 7
    exp['train_batch_size'] = [8]
    exp['val_batch_size'] = [8]
    exp['epochs'] = [2048 * 2]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'image_float32',
        'label_float32',
        'image_to_bgr',  # also flips bgr to rgb
        # 'lr_flip_image_label',
        # 'rot_image_label',
        'ilsvrc12_normalize',
        # 'bsds_normalize',
        'coco_labels',
        'res_nn_image_label',
        # 'res_image_label',
        # 'uint8_rescale',
        # 'zero_one',
        # 'bfloat16',
    ]]
    exp['val_augmentations'] = [[
        # 'bsds_mean',
        # 'bsds_normalize',
        'image_float32',
        'label_float32',
        'image_to_bgr',  # also flips bgr to rgb
        'ilsvrc12_normalize',
        # 'cc_image_label',
        'coco_labels',
        'res_nn_image_label',
        # 'uint8_rescale',
        # 'zero_one',
    ]]
    return exp

