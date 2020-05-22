#!/usr/bin/env python
import tensorflow as tf
from layers.recurrent import hgru_bn_cd as hgru


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
    alpha = 5.
    beta = 1.
    dist = tf.distributions.Beta(alpha, beta)
    for idx, acts in enumerate(zip(layers)):
        (xs, inhs) = acts[0]
        # xs *= mask
        # inhs *= tf.abs(1 - mask)
        rand = dist.sample([])
        sel = tf.cast(
            tf.round(
                tf.random_uniform(
                    [], minval=1, maxval=bs - 1)), tf.int32)
        xs = tf.stop_gradient(rand * xs + ((1 - rand) * tf.roll(xs, sel, 0)))
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
