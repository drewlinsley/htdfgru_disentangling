#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import normalization
from layers.recurrent import gammanet_refactored as gammanet
from ops import tf_fun


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
    data_tensor, long_data_format = tf_fun.interpret_data_format(
        data_tensor=data_tensor,
        data_format=data_format)
    normalization_type = 'instance_norm'

    # Prepare gammanet structure

    # Modules
    # Inputs (list of tensor names) /outputs (feature maps)
    # Pool or upsample

    compression = ['pool', 'pool', 'embedding', 'upsample', 'upsample']
    ff_kernels = [[False]] + [[3, 3]] * (len(compression) - 1)
    ff_repeats = [False, 4, 4, False, False]
    features = [24, 32, 48, 32, 24]
    fgru_kernels = [[9, 9], [3, 3], [3, 3], [1, 1], [1, 1]]
    gammanet_constructor = tf_fun.get_gammanet_constructor(
        compression=compression,
        ff_kernels=ff_kernels,
        ff_repeats=ff_repeats,
        features=features,
        fgru_kernels=fgru_kernels)

    # Additional gammanet options
    aux = {
        'attention': 'se',  # 'gala',  # 'gala',  # 'gala', se
        'attention_layers': 2,
        'saliency_filter': 5,
        'use_homunculus': False,
        'ff_nl': tf.nn.elu,
        'fgru_connectivity': 'full',
        'upsample_convs': True,
        'separable_convs': 4,  # Multiplier
        'separable_upsample': True,
        'residual': True,  # intermediate resid connections
        'while_loop': False,
        'skip': False,
        'nonnegative': False,
        'symmetric_weights': True,
        'force_alpha_divisive': True,
        'force_omega_nonnegative': True,
        'td_cell_state': False,
        'td_gate': False,  # Add top-down activity to the in-gate
    }
    with tf.variable_scope('cnn', reuse=reuse):

        # Add fixed input conv layer
        in_emb = tf.layers.conv2d(
            inputs=data_tensor,
            filters=16,
            kernel_size=7,
            strides=(1, 1),
            padding='same',
            data_format=long_data_format,
            activation=tf.nn.relu,
            trainable=training,
            use_bias=True,
            name='l0')
        in_emb = normalization.apply_normalization(
            activity=in_emb,
            name='input_norm',
            normalization_type=normalization_type,
            data_format=data_format,
            training=training,
            reuse=reuse)

        # Run fGRU
        gammanet_layer = gammanet.GN(
            layer_name='fgru',
            x=in_emb,
            gammanet_constructor=gammanet_constructor,
            data_format=data_format,
            reuse=reuse,
            timesteps=8,
            readout=[len(gammanet_constructor), 2],
            strides=[1, 1, 1, 1],
            fgru_normalization_type=normalization_type,
            ff_normalization_type=normalization_type,
            horizontal_padding='SAME',
            ff_padding='SAME',
            recurrent_ff=False,
            horizontal_kernel_initializer=tf.initializers.orthogonal(),
            kernel_initializer=tf.initializers.orthogonal(),
            aux=aux,
            pool_strides=[2, 2],
            pooling_kernel=[4, 4],
            up_kernel=[4, 4],
            train=training)
        # h2 = gammanet_layer(in_emb)
        h2, h_deep = gammanet_layer(in_emb)

    with tf.variable_scope('cv_readout', reuse=reuse):
        # Apply one more normalization
        h2 = normalization.apply_normalization(
            activity=h2,
            name='output_norm',
            normalization_type=normalization_type,
            data_format=data_format,
            training=training,
            reuse=reuse)
        # h_deep = apply_normalization(
        #     activity=h_deep,
        #     name='output_normh_deep',
        #     normalization_type=normalization_type,
        #     data_format=data_format,
        #     training=training,
        #     reuse=reuse)

        # Then read out
        activity = tf.layers.conv2d(
            inputs=h2,
            filters=output_shape,
            kernel_size=(1, 1),
            padding='same',
            data_format=long_data_format,
            name='pre_readout_conv',
            use_bias=True,
            reuse=reuse)
        # h_deep = tf.layers.conv2d(
        #     inputs=h_deep,
        #     filters=output_shape,
        #     kernel_size=(1, 1),
        #     padding='same',
        #     data_format=long_data_format,
        #     name='pre_readout_conv1',
        #     use_bias=True,
        #     reuse=reuse)
    if long_data_format is 'channels_first':
        activity = tf.transpose(activity, (0, 2, 3, 1))
    extra_activities = {
        'activity_low': h2,
        'activity_high': h_deep,
    }
    if activity.dtype != tf.float32:
        activity = tf.cast(activity, tf.float32)
    # return [activity, h_deep], extra_activities
    return activity, extra_activities
