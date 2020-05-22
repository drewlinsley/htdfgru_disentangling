#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import normalization, conv
from layers.recurrent import hgru_bn_cd as hgru
import numpy as np


def gaussian(x, y, x0, y0, sigma=1):
    """Add gaussian to tensorflow image."""
    gx = tf.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    gy = tf.exp(-(y - y0) ** 2 / (2 * sigma ** 2))
    # np.outer(gx, gy)
    g = tf.einsum("i,j->ij", gx, gy)
    g /= tf.reduce_sum(g)
    return tf.expand_dims(tf.expand_dims(g, 0), -1)
    # x0 = tf.random_uniform([], minval=0.4, maxval=0.6)
    # y0 = tf.random_uniform([], minval=0.4, maxval=0.6)
    # sigma = tf.random_uniform([], minval=0.0, maxval=0.0)
    # mask = gaussian(
    #     np.linspace(0, 1, x.get_shape().as_list()[1]),
    #     np.linspace(0, 1, x.get_shape().as_list()[2]),
    #     x0=x0,
    #     y0=y0,
    #     sigma=sigma)
    # mask = tf.cast(tf.greater(mask, tf.reduce_mean(mask)), tf.float32)


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    alpha = 5.
    beta = 1.
    bs = data_tensor.get_shape().as_list()[0]
    dist = tf.distributions.Beta(alpha, beta)
    rand = dist.sample([])
    sel = tf.cast(
        tf.round(
            tf.random_uniform(
                [], minval=1,
                maxval=bs - 1)), tf.int32)
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('input', reuse=reuse):
            o_x = tf.layers.conv2d(
                inputs=data_tensor,
                filters=24,
                kernel_size=11,
                name='l0',
                strides=(1, 1),
                padding='same',
                activation=tf.nn.relu,
                trainable=training,
                use_bias=True)
            x = rand * o_x + ((1 - rand) * tf.roll(o_x, sel, 0))
            layer_hgru = hgru.hGRU(
                'hgru_1',
                x_shape=x.get_shape().as_list(),
                timesteps=8,
                h_ext=15,
                strides=[1, 1, 1, 1],
                padding='SAME',
                aux={'reuse': False, 'constrain': False},
                train=training)
            h2, inh = layer_hgru.build(x)

    with tf.variable_scope('ro', reuse=reuse):
        h2 = normalization.batch(
            bottom=h2,
            renorm=False,
            name='hgru_bn',
            training=training)
        activity = conv.readout_layer(
            activity=h2,
            reuse=reuse,
            training=training,
            output_shape=output_shape)
    extra_activities = {
        'activity': h2
    }
    return activity, extra_activities

