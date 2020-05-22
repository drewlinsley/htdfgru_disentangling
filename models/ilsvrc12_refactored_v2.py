#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import normalization
from layers.feedforward import conv
from layers.recurrent import gammanet_refactored as gammanet
from ops import tf_fun


# def l1():
#     return 1e-4


def get_aux():
    """Auxilary options for GN."""
    return {
        'attention': 'gala',  # 'gala',  # 'gala',  # 'se',  # 'gala',  # 'gala',  # 'gala', se
        'attention_layers': 2,
        'norm_attention': False,
        'saliency_filter': 5,
        'use_homunculus': False,
        'gate_homunculus': False,
        'single_homunculus': False,
        'upsample_nl': True,
        'upsample_convs': True,
        'separable_upsample': False,
        'separable_convs': False,  # Multiplier
        'combine_fgru_output': True,
        # 'fgru_output_normalization': True,
        'fgru_output_normalization': False,
        'fgru_batchnorm': True,
        'skip_connections': True,
        'residual': True,  # intermediate resid connections
        'while_loop': False,
        'image_resize': tf.image.resize_nearest_neighbor,  # resize_nearest_neighbor, 'unpool',
        'bilinear_init': False,
        'nonnegative': True,
        'adaptation': False,
        # 'symmetric_weights': 'spatial_channel',  #  'spatial_channel',  # 'channel',  # 'spatial_channel',  # 'spatial_channel',
        'symmetric_weights': 'channel',  #  'spatial_channel',  # 'channel',  # 'spatial_channel',  # 'spatial_channel',
        'force_alpha_divisive': False,
        'force_omega_nonnegative': False,
        'td_cell_state': False,
        'td_gate': False,  # Add top-down activity to the in-gate
    }


def v2_big_working():
    compression = ['pool', 'pool', 'pool', 'pool', 'embedding', 'upsample', 'upsample', 'upsample', 'upsample']
    ff_kernels = [[False], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    # ff_repeats = [False, 1, 1, 1, 1, 1, 1]  # 4
    ff_repeats = [False, 1, 1, 1, 1, 1, 1, 1, 1]  # 4
    features = [14, 16, 18, 48, 64, 48, 18, 16, 14]  # DEFAULT
    # fgru_kernels = [[15, 15], [13, 13], [11, 11], [9, 9], [1, 1], [1, 1], [1, 1]]
    fgru_kernels = [[9, 9], [7, 7], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    # fgru_kernels = [[9, 9], [3, 3], [3, 3], [3, 3], [1, 1], [1, 1], [1, 1]]
    # fgru_kernels = [[9, 9], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    ar = ['fgru_3']  # Output layer ids
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
    # normalization_type = 'no_param_instance_norm'
    # output_normalization_type = 'instance_norm'
    fgru_normalization_type = 'no_param_batch_norm_original'
    ff_normalization_type = 'no_param_batch_norm_original'
    output_normalization_type = 'batch_norm_original'
    ff_kernel_size = (5, 5)
    ff_nl = tf.nn.relu
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
        activity = tf.layers.conv2d(
            inputs=data_tensor,
            filters=gammanet_constructor[0]['features'],
            kernel_size=ff_kernel_size,
            padding='same',
            # strides=(2, 2),
            data_format=long_data_format,
            name='l0_0',
            activation=ff_nl,
            use_bias=True,
            reuse=reuse)
        activity = normalization.apply_normalization(
            activity=activity,
            name='input_norm',
            normalization_type=output_normalization_type,
            data_format=data_format,
            training=training,
            trainable=training,
            reuse=reuse)  # + pre_activity
        activity = tf.layers.conv2d(
            inputs=activity,
            filters=gammanet_constructor[0]['features'],
            kernel_size=ff_kernel_size,
            padding='same',
            # strides=(2, 2),
            data_format=long_data_format,
            name='l0_1',
            activation=ff_nl,
            use_bias=True,
            reuse=reuse)

        # Shift conv-layers to the front
        # Run fGRU
        gn = gammanet.GN(
            layer_name='fgru',
            gammanet_constructor=gammanet_constructor,
            data_format=data_format,
            reuse=reuse,
            timesteps=3,
            fgru_connectivity='',  # 'all_to_all',
            additional_readouts=additional_readouts,
            fgru_normalization_type=fgru_normalization_type,
            ff_normalization_type=ff_normalization_type,
            horizontal_padding='SAME',
            ff_padding='SAME',
            ff_nl=ff_nl,
            stop_loop='fgru_3',
            recurrent_ff=True,
            horizontal_kernel_initializer=tf.initializers.orthogonal(),
            kernel_initializer=tf.initializers.orthogonal(),
            gate_initializer=tf.initializers.orthogonal(),
            # horizontal_kernel_initializer=tf.contrib.layers.xavier_initializer(),
            # gate_initializer=tf.contrib.layers.xavier_initializer(),
            # gate_initializer=tf.contrib.layers.xavier_initializer(),
            aux=aux,
            strides=[1, 1, 1, 1],
            pool_strides=[2, 2],
            pool_kernel=[2, 2],
            up_kernel=[4, 4],
            train=training)
        h2, h_deep = gn(X=activity)

    with tf.variable_scope('cv_readout', reuse=reuse):
        # Apply one more normalization
        # activity = tf.layers.max_pooling2d(
        #     inputs=h2,
        #     pool_size=(2, 2),
        #     strides=(2, 2),
        #     padding='same',
        #     data_format=long_data_format,
        #     name='output_pool_3')
        activity = normalization.apply_normalization(
            activity=h2,
            name='output_norm_activity',
            normalization_type=output_normalization_type,
            data_format=data_format,
            training=training,
            trainable=training,
            reuse=reuse)  # + pre_activity

        # Then read out
        activity = tf.layers.conv2d(
            inputs=activity,
            filters=2048,
            kernel_size=ff_kernel_size,
            padding='same',
            # strides=(2, 2),
            data_format=long_data_format,
            name='readout_fc_0',
            activation=ff_nl,
            use_bias=True,
            reuse=reuse)
        activity = normalization.apply_normalization(
            activity=activity,
            name='readout_norm_activity',
            normalization_type=output_normalization_type,
            data_format=data_format,
            training=training,
            trainable=training,
            reuse=reuse)  # + pre_activity
        activity = tf.reduce_mean(activity, reduction_indices=[1, 2])
        activity = tf.layers.dense(
            inputs=activity,
            units=output_shape,
            activation=None,
            use_bias=True,
            bias_initializer=tf.constant_initializer(1. / 1000.),
            name='readout_fc_2',
            reuse=reuse)
    if long_data_format is 'channels_first':
        activity = tf.transpose(activity, (0, 2, 3, 1))
    extra_activities = {
        # 'activity_low': h2,
        # 'activity_high': h_deep,
    }
    if activity.dtype != tf.float32:
        activity = tf.cast(activity, tf.float32)
    # return [activity, h_deep], extra_activities
    return activity, extra_activities
