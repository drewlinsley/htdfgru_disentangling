#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from layers.feedforward import normalization
from layers.recurrent import hgru_bn_cd as hgru


def gaussian(x, y, x0, y0, sigma=1):
    """Add gaussian to tensorflow image."""
    gx = tf.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    gy = tf.exp(-(y - y0) ** 2 / (2 * sigma ** 2))
    # np.outer(gx, gy)
    g = tf.einsum("i,j->ij", gx, gy)
    g /= tf.reduce_sum(g)
    return tf.expand_dims(tf.expand_dims(g, 0), -1)


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('input', reuse=reuse):
            x = tf.layers.conv2d(
                inputs=data_tensor,
                filters=24,
                kernel_size=11,
                name='l0',
                strides=(1, 1),
                padding='same',
                activation=tf.nn.relu,
                trainable=training,
                use_bias=True)
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

    layers = [[x, inh]]
    ds_list = []
    bs = data_tensor.get_shape().as_list()[0]
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
    alpha = 0.4
    dist = tf.distributions.Beta(alpha, alpha)
    for idx, acts in enumerate(zip(layers)):
        (xs, inhs) = acts[0]
        # xs *= mask
        # inhs *= tf.abs(1 - mask)
        rand = dist.sample([])
        sel = tf.cast(tf.round(tf.random_uniform([], minval=1, maxval=bs - 1)), tf.int32)
        xs = rand * xs + ((1 - rand) * tf.roll(xs, sel, 0))
        xs = tf.reshape(xs, [bs, -1])
        inhs = tf.reshape(inhs, [bs, -1])

        # Augmentation
        rolled_xs = []
        for r in range(1, bs - 1):
            rolled_xs += [tf.roll(xs, r, 0)]
        xs = tf.concat([xs] + rolled_xs, 0)
        inhs = tf.tile(inhs, [bs - 1, 1])

        # CPC distances
        # xs = tf.nn.l2_normalize(xs, -1)
        # inhs = tf.nn.l2_normalize(inhs, -1)
        ds = tf.reduce_sum((xs * inhs), axis=-1)
        ds = tf.reshape(ds, [bs - 1, -1])
        ds_list += [ds]
    ds_list = tf.transpose(tf.concat(ds_list, -1))
    return ds_list, {}
