#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.recurrent import hgru_bn_for as hgru


def laplace():
    """Add this laplace regularization to the model."""
    return 1e-4


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        # Add input
        so_filters = np.load(
            os.path.join('weights', 'so_filters.npy')).squeeze().reshape(
                11, 11, 3, 8 * 4)[..., 0:32:4]
        so_filter_tensor = tf.get_variable(
            name='so_filters',
            initializer=so_filters,
            trainable=training)
        so_bias = tf.get_variable(
            name='so_bias',
            initializer=tf.zeros(so_filters.shape[-1]),
            trainable=training)
        in_emb = tf.nn.conv2d(
            input=data_tensor,
            filter=so_filter_tensor,
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='so')
        in_emb = tf.nn.bias_add(in_emb, so_bias)
        # in_emb = in_emb ** 2
        layer_hgru = hgru.hGRU(
            'hgru_1',
            x_shape=in_emb.get_shape().as_list(),
            timesteps=8,
            h_ext=21,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux={
                'reuse': False,
                'constrain': False,
                'nonnegative': True,
                'while_loop': False,
                'horizontal_dilations': [1, 2, 2, 1]
            },
            train=training)
        h2 = layer_hgru.build(in_emb)
        h2 = normalization.batch(
            bottom=h2,
            renorm=False,
            name='hgru_bn',
            training=training)
        activity = conv.readout_layer(
            activity=h2,
            reuse=reuse,
            training=training,
            pool_type='max',  # 'select',
            output_shape=output_shape,
            features=so_filters.shape[-1])
    extra_activities = {
        'activity': h2
    }
    return activity, extra_activities
