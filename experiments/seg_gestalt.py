import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 5  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        # 'seg_cluttered_nist_3_ix2v2_50k_real',
        'curv_contour_segments_length_14_full_real',
    ]
    exp['val_dataset'] = [
        # 'seg_cluttered_nist_3_ix2v2_50k_real',
        'curv_contour_segments_length_14_full_real',
    ]
    exp['model'] = [
        # 'resnet_18',
        # 'resnet_50',
        # 'resnet_152',
        # 'unet',
        # 'seung_unet',
        # 'feedback_hgru',
        # 'feedback_hgru_mely',
        # 'feedback_hgru_fs',
        # 'feedback_hgru_fs_mely',
        # 'hgru_bn'
        'bu_fgru',
        'h_fgru',
        'td_fgru_t1',
        'td_fgru_t1_skip'
    ]
    exp['validation_period'] = [1000]
    exp['validation_steps'] = [625]
    exp['shuffle_val'] = [True]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]

    # Model hyperparameters
    exp['lr'] = [1e-6]  # 1e-3
    exp['exclusion_lr'] = 1e-3  # 1e-5
    exp['exclusion_scope'] = 'readout'
    exp['optimizer'] = ['adam']
    exp['loss_function'] = ['bi_bce']
    exp['val_loss_function'] = ['bi_bce']
    exp['score_function'] = ['pixel_error']
    exp['optimizer'] = ['nadam']  # , 'adam']
    exp['train_batch_size'] = [4]
    exp['val_batch_size'] = [4]
    exp['epochs'] = [64]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'grayscale',
        'res_image_label',
        # 'left_right',
        # 'up_down',
        'stack3d',
        'pascal_normalize',
        'image_to_bgr',
        'threshold_label_255',
    ]]
    exp['val_augmentations'] = [[
        'grayscale',
        'res_image_label',
        # 'left_right',
        # 'up_down',
        'stack3d',
        'pascal_normalize',
        'image_to_bgr',
        'threshold_label_255',
    ]]
    return exp

