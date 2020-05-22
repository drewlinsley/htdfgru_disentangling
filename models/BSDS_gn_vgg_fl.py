#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import normalization
from layers.feedforward import conv
from layers.recurrent import recurrent_vgg16_fl as vgg16
from layers.recurrent import gammanet_refactored as gammanet
from ops import tf_fun


# def l1():
#     return 1e-4


def get_aux():
    """Auxilary options for GN."""
    return {
        'attention': 'gala',  # 'gala', 'se', False
        'attention_layers': 1,
        'norm_attention': False,
        'saliency_filter': 5,
        # 'gate_nl': tf.keras.activations.hard_sigmoid,
        'use_homunculus': False,
        'gate_homunculus': False,
        'single_homunculus': False,
        'combine_fgru_output': False,
        'upsample_nl': True,
        'upsample_convs': True,
        'separable_upsample': False,
        'separable_convs': False,  # Multiplier
        # 'fgru_output_normalization': True,
        'fgru_output_normalization': False,
        'fgru_batchnorm': True,
        'skip_connections': False,
        'residual': True,  # intermediate resid connections
        'while_loop': False,
        'image_resize': tf.image.resize_nearest_neighbor,
        'bilinear_init': False,
        'nonnegative': True,
        'adaptation': False,
        'symmetric_weights': 'channel',  # 'spatial_channel', 'channel', False
        'force_alpha_divisive': False,
        'force_omega_nonnegative': False,
        'td_cell_state': False,
        'td_gate': False,  # Add top-down activity to the in-gate
        # 'dilations': [1, 2, 2, 1],
        'partial_padding': False
    }


def v2_big_working():
    compression = ['pool', 'embedding', 'upsample']
    ff_kernels = [[False]] * len(compression)
    ff_repeats = [[False]] * len(compression)  # [False, 1, 1, 1, 1, 1, 1, 1, 1]  # 4
    features = [64, 512, 64]  # Default
    # features = [24, 18, 18, 48, 64, 48, 18, 18, 24]  # Bottleneck
    # features = [16, 18, 20, 48, 20, 18, 16]  # DEFAULT
    fgru_kernels = [[9, 9], [1, 1], [1, 1]]
    ar = ['']  # , 'fgru_3', 'fgru_4']  # Output layer ids
    # ar = ['fgru_4']  # Output layer ids 
    return compression, ff_kernels, ff_repeats, features, fgru_kernels, ar


def build_model(
        data_tensor,
        reuse,
        training,
        output_shape,
        data_format='NHWC'):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[-1]
    elif isinstance(output_shape, dict):
        output_shape = output_shape['output']
    input_training = training   # training
    readout_training = training  # False
    norm_moments_training = training  # Force instance norm
    norm_params_training = training
    fgru_kernel_training = training
    fgru_param_training = training
    ff_gate_training = training
    fgru_gate_training = training
    remaining_param_training = training
    normalization_type = 'no_param_batch_norm_original'
    output_normalization_type = 'batch_norm_original'
    # normalization_type = 'no_param_instance_norm'
    # output_normalization_type = 'instance_norm'
    ff_kernel_size = (3, 3)
    ff_nl = tf.nn.elu
    data_tensor, long_data_format = tf_fun.interpret_data_format(
        data_tensor=data_tensor,
        data_format=data_format)

    # Prepare gammanet structure
    (
        compression,
        ff_kernels,
        ff_repeats,
        features,
        fgru_kernels,
        additional_readouts) = v2_big_working()
    gammanet_constructor = tf_fun.get_gammanet_constructor(
        compression=compression,
        ff_kernels=ff_kernels,
        ff_repeats=ff_repeats,
        features=features,
        fgru_kernels=fgru_kernels)
    aux = get_aux()

    # Build model
    with tf.variable_scope('gammanet', reuse=reuse):
        aux = get_aux()
        vgg = vgg16.Vgg16(
            vgg16_npy_path='/media/data_cifs/clicktionary/pretrained_weights/vgg16.npy',
            reuse=reuse,
            aux=aux,
            train=training,
            timesteps=4,
            fgru_normalization_type=normalization_type,
            ff_normalization_type=normalization_type)
        vgg(rgb=data_tensor, constructor=gammanet_constructor)
        # h2_rem = [vgg.conv5_3, vgg.conv4_3, vgg.conv3_3, vgg.conv2_2, vgg.conv1_1]
        h2 = vgg.fgru_0

    with tf.variable_scope('contour_readout', reuse=reuse):
        # Apply one more normalization
        # h_deep = apply_normalization(
        #     activity=h_deep,
        #     name='output_normh_deep',
        #     normalization_type=normalization_type,
        #     data_format=data_format,
        #     training=training,
        #     reuse=reuse)
        if 0:
            hs = []
            for h in h2_rem:
                hs += [tf.image.resize_nearest_neighbor(h, data_tensor.get_shape().as_list()[1:3])]
            activity = normalization.apply_normalization(
                activity=tf.concat(hs, axis=-1),
                name='output_norm',
                normalization_type=output_normalization_type,
                data_format=data_format,
                training=norm_moments_training,
                trainable=training,
                reuse=reuse)
        else:
            activity = normalization.apply_normalization(
                activity=h2,
                name='output_norm',
                normalization_type=output_normalization_type,
                data_format=data_format,
                training=norm_moments_training,
                trainable=training,
                reuse=reuse)

        # Then read out
        activity = tf.layers.conv2d(
            inputs=activity,
            filters=output_shape,
            kernel_size=(1, 1),
            padding='same',
            data_format=long_data_format,
            name='readout_conv',
            activation=None,
            trainable=training,
            use_bias=True,
            reuse=reuse)
    if long_data_format is 'channels_first':
        activity = tf.transpose(activity, (0, 2, 3, 1))
    extra_activities = {
    }
    if activity.dtype != tf.float32:
        activity = tf.cast(activity, tf.float32)
    # return [activity, h_deep], extra_activities
    return activity, extra_activities

