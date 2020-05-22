#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.recurrent import constrained_h_td_fgru_v5_while as gammanet
from collections import OrderedDict


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
    if data_format is 'NCHW':
        data_tensor = tf.transpose(data_tensor, (0, 3, 1, 2))
        long_data_format = 'channels_first'
    else:
        long_data_format = 'channels_last'

    with tf.variable_scope('cnn', reuse=reuse):
        normalization_type = 'instance_norm'  #
        # # Concatenate standard deviation
        # _, var = tf.nn.moments(data_tensor, axes=[3])
        # std = tf.expand_dims(tf.sqrt(var), axis=-1)
        # data_tensor = tf.concat([data_tensor, std], axis=-1)

        # Add input
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

        # Run fGRU
        hgru_kernels = OrderedDict()
        hgru_kernels['h1'] = [9, 9]  # height/width
        hgru_kernels['h2'] = [3, 3]
        hgru_kernels['fb1'] = [1, 1]
        hgru_features = OrderedDict()
        hgru_features['h1'] = [16, 16]  # Fan-in/fan-out, I and E (match fb1)
        hgru_features['h2'] = [48, 48]
        hgru_features['fb1'] = [16, 16]  # (match h1)
        # intermediate_ff = [16, 16, 24, 24, 32, 32]  # Last feature must match h2
        # intermediate_ks = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        # intermediate_repeats = [4, 4, 4, 4, 4, 4]  # Repeat each FF this many times
        intermediate_ff = [16, 48]  # Last feature must match h2
        intermediate_ks = [[3, 3], [3, 3]]
        intermediate_repeats = [4, 4]  # Repeat each FF this many times
        gammanet_layer = gammanet.GN(
            layer_name='fgru',
            x=in_emb,
            data_format=data_format,
            reuse=reuse,
            timesteps=2,
            strides=[1, 1, 1, 1],
            hgru_features=hgru_features,
            hgru_kernels=hgru_kernels,
            intermediate_ff=intermediate_ff,
            intermediate_ks=intermediate_ks,
            intermediate_repeats=intermediate_repeats,
            horizontal_kernel_initializer=lambda x: tf.initializers.orthogonal(gain=0.1),
            kernel_initializer=lambda x: tf.initializers.orthogonal(gain=0.1),
            padding='SAME',
            aux={
                'readout': 'fb',
                'attention': 'se',  # 'gala',  # 'gala', se
                'attention_layers': 2,
                'saliency_filter': 7,
                'use_homunculus': True,
                'upsample_convs': True,
                'separable_convs': 4,  # Multiplier
                'separable_upsample': True,
                'td_cell_state': True,
                'td_gate': False,  # Add top-down activity to the in-gate
                'normalization_type': normalization_type,
                'residual': True,  # intermediate resid connections
                'while_loop': False,
                'skip': False,
                'symmetric_weights': True,
                'bilinear_init': True,
                'include_pooling': True,
                'time_homunculus': True,

                'squeeze_fb': False,  # Compress Inh-hat with a 1x1 conv
                'force_horizontal': False,
                'time_skips': False,
                'timestep_output': False,
                'excite_se': False,  # Add S/E in the excitation stage
            },
            pool_strides=[2, 2],
            pooling_kernel=[4, 4],  # 4, 4 helped but also try 3, 3
            up_kernel=[4, 4],
            train=training)
        h2 = gammanet_layer(in_emb)
        if normalization_type is 'batch_norm':
            h2 = normalization.batch_contrib(
                bottom=h2,
                renorm=False,
                name='hgru_bn',
                dtype=h2.dtype,
                data_format=data_format,
                training=training)
        elif normalization_type is 'instance_norm':
            h2 = normalization.instance(
                bottom=h2,
                data_format=data_format,
                training=training)
        elif normalization_type is 'ada_batch_norm':
            h2 = normalization.batch_contrib(
                bottom=h2,
                renorm=False,
                name='hgru_bn',
                dtype=h2.dtype,
                data_format=data_format,
                training=training)
        else:
            raise NotImplementedError(normalization_type)
    with tf.variable_scope('cv_readout', reuse=reuse):
        activity = tf.layers.conv2d(
            inputs=h2,
            filters=output_shape,
            kernel_size=(1, 1),
            padding='same',
            data_format=long_data_format,
            name='pre_readout_conv',
            use_bias=True,
            reuse=reuse)
    if long_data_format is 'channels_first':
        activity = tf.transpose(activity, (0, 2, 3, 1))
    extra_activities = {
        'activity': h2
    }
    if activity.dtype != tf.float32:
        activity = tf.cast(activity, tf.float32)
    return activity, extra_activities
