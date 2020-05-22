#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.recurrent import constrained_h_td_fgru_v5 as hgru
from collections import OrderedDict


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    elif isinstance(output_shape, dict):
        output_shape = output_shape['output']
    with tf.variable_scope('cnn', reuse=reuse):
        normalization_type = 'ada_batch_norm'  # 'instance_norm'
        # # Concatenate standard deviation
        # _, var = tf.nn.moments(data_tensor, axes=[3])
        # std = tf.expand_dims(tf.sqrt(var), axis=-1)
        # data_tensor = tf.concat([data_tensor, std], axis=-1)

        # Add input
        in_emb = conv.skinny_input_layer(
            X=data_tensor,
            reuse=reuse,
            training=training,
            features=16,
            conv_activation=tf.nn.relu,
            conv_kernel_size=7,
            pool=False,
            name='l0')

        # Run fGRU
        hgru_kernels = OrderedDict()
        hgru_kernels['h1'] = [15, 15]  # height/width
        hgru_kernels['h2'] = [5, 5]
        hgru_kernels['fb1'] = [1, 1]
        hgru_features = OrderedDict()
        hgru_features['h1'] = [16, 16]  # Fan-in/fan-out, I and E (match fb1)
        hgru_features['h2'] = [48, 48]
        hgru_features['fb1'] = [16, 16]  # (match h1)
        # hgru_features['fb1'] = [24, 12]  # (match h1 unless squeeze_fb)
        intermediate_ff = [24, 24, 48]  # Last feature must match h2
        intermediate_ks = [[5, 5], [5, 5], [5, 5]]
        intermediate_repeats = [1, 1, 1]  # Repeat each interm this many times
        layer_hgru = hgru.hGRU(
            'fgru',
            x_shape=in_emb.get_shape().as_list(),
            timesteps=8,
            strides=[1, 1, 1, 1],
            hgru_features=hgru_features,
            hgru_kernels=hgru_kernels,
            intermediate_ff=intermediate_ff,
            intermediate_ks=intermediate_ks,
            intermediate_repeats=intermediate_repeats,
            padding='SAME',
            aux={
                'readout': 'fb',
                'squeeze_fb': False,  # Compress Inh-hat with a 1x1 conv
                'td_gate': True,  # Add top-down activity to the in-gate
                'attention': 'gala',  # 'gala',
                'attention_layers': 2,
                'upsample_convs': True,
                'td_cell_state': True,
                'normalization_type': normalization_type,
                'excite_se': False,  # Add S/E in the excitation stage
                'residual': True,  # intermediate resid connections
                'while_loop': False,
                'skip': True,
                'time_skips': False,
                'force_horizontal': False,
                'symmetric_weights': True,
                'timestep_output': False,
                'bilinear_init': True,
                'include_pooling': True
            },
            pool_strides=[2, 2],
            pooling_kernel=[4, 4],
            up_kernel=[4, 4],
            train=training)
        h2 = layer_hgru.build(in_emb)
        if normalization_type is 'batch_norm':
            h2 = normalization.batch(
                bottom=h2,
                renorm=False,
                name='hgru_bn',
                training=training)
        elif normalization_type is 'instance_norm':
            h2 = normalization.instance(
                bottom=h2,
                training=training)
        elif normalization_type is 'ada_batch_norm':
            h2 = normalization.batch(
                bottom=h2,
                renorm=False,
                name='hgru_bn',
                training=True)
        else:
            raise NotImplementedError(normalization_type)
        fc = tf.layers.conv2d(
            inputs=h2,
            filters=output_shape,
            kernel_size=1,
            name='fc')
        # activity = tf.reduce_mean(fc, reduction_indices=[1, 2])
        activity = tf.reduce_max(fc, reduction_indices=[1, 2])
    extra_activities = {
        'activity': h2
    }
    return activity, extra_activities
