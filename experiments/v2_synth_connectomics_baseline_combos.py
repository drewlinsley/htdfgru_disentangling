import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 10  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'v2_synth_connectomics_baseline_1',
        # 'v2_synth_connectomics_baseline_5',
        # 'v2_synth_connectomics_baseline_10',
        # 'v2_synth_connectomics_baseline_20',
        # 'v2_synth_connectomics_baseline_200',
    ]
    exp['val_dataset'] = [
        # 'v2_synth_connectomics_baseline_200',
        'v2_synth_connectomics_baseline_20',
        # 'v2_synth_connectomics_size',
        # 'v2_synth_connectomics_lumcontrast',
    ]
    exp['model'] = [
        # 'seung_unet_per_pixel',
        # 'seung_unet_per_pixel_wd',
        # 'seung_unet_per_pixel_param_ctrl',
        # 'gammanet_t8_per_pixel',
        'gammanet_t8_per_pixel_v5',
        # 'gammanet_t4_per_pixel',
        # 'gammanet_t1_per_pixel',
        # 'gammanet_t8_per_pixel_skips',
        # 'gammanet_t8_per_pixel_lesion_mult',
        # 'gammanet_t8_per_pixel_lesion_add',
        # 'hgru_bn_per_pixel',
        # 'gru_t8_per_pixel',
        # 'lstm_t8_per_pixel',
        # 'seung_unet_per_pixel_param_ctrl_rn_2'
        # 'gammanet_t8_per_pixel_disinhibition',
        # 'gru_v2_t8_per_pixel',
        # 'lstm_v2_t8_per_pixel',
        # 'ffn_per_pixel',
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
    exp['loss_function'] = ['sparse_cce_image']  # ['timestep_sparse_ce_image']
    exp['score_function'] = ['f1']  # ['timestep_f1']
    exp['optimizer'] = ['nadam']  # , 'adam']
    exp['train_batch_size'] = [5]
    exp['val_batch_size'] = [5]
    exp['epochs'] = [1000]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'singleton',
        'sgl_label',
        'res_image_label',
        # 'left_right',
        # 'up_down',
        'uint8_rescale',
        'zero_one',
        'threshold_label'
    ]]
    exp['val_augmentations'] = [[
        'singleton',
        'sgl_label',
        'res_image_label',
        # 'left_right',
        # 'up_down',
        'uint8_rescale',
        'zero_one',
        'threshold_label'
    ]]
    return exp
