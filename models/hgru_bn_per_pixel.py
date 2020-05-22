#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.recurrent import hgru_bn_for as hgru


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        # Add input
        in_emb = conv.skinny_input_layer(
            X=data_tensor,
            reuse=reuse,
            training=training,
            features=24,
            conv_activation=tf.nn.elu,
            conv_kernel_size=7,
            pool=False,
            name='l0')
        layer_hgru = hgru.hGRU(
            'hgru_1',
            x_shape=in_emb.get_shape().as_list(),
            timesteps=8,
            h_ext=9,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux={'reuse': False, 'constrain': False, 'nonnegative': True},
            train=training)
        h2 = layer_hgru.build(in_emb)
        h2 = normalization.batch(
            bottom=h2,
            renorm=True,
            name='hgru_bn',
            training=training)
        activity = conv.conv_layer(
            bottom=h2,
            name='pre_readout_conv',
            num_filters=output_shape['output'],
            kernel_size=1,
            trainable=training,
            use_bias=True)
    extra_activities = {
    }
    return activity, extra_activities
