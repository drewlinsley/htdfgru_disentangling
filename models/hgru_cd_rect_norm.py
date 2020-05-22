#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import normalization
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
            # h2 = normalization.batch(
            #     bottom=h2,
            #     renorm=True,
            #     name='hgru_bn',
            #     training=training)
            # inh = normalization.batch(
            #     bottom=inh,
            #     renorm=True,
            #     name='hgru_inh',
            #     training=training)

    layers = [[x, tf.nn.relu(inh)]]
    ds_list = []
    bs = data_tensor.get_shape().as_list()[0]
    for idx, acts in enumerate(zip(layers)):
        (xs, inhs) = acts[0]
        xs = tf.reshape(xs, [bs, -1])
        inhs = tf.reshape(inhs, [bs, -1])

        # Augmentation
        # roll_choices = tf.range(
        #     bs - 1,
        #     dtype=tf.int32)
        # samples = tf.multinomial(
        #     tf.log([tf.ones_like(tf.cast(roll_choices, tf.float32))]), 1)
        # rand_roll = roll_choices[tf.cast(samples[0][0], tf.int32)]
        # rolled_xs = tf.roll(xs, rand_roll, 0)
        # xs = tf.concat([xs, rolled_xs], 0)
        # inhs = tf.concat([inhs, inhs], 0)
        rolled_xs = []
        for r in range(1, bs - 1):
            rolled_xs += [tf.roll(xs, r, 0)]
        xs = tf.concat([xs] + rolled_xs, 0)
        inhs = tf.tile(inhs, [bs - 1, 1])

        # CPC distances
        # denom_xs = tf.sqrt(
        #     tf.reduce_sum(tf.matmul(xs, xs, transpose_b=True), axis=-1))
        # denom_inhs = tf.sqrt(
        #     tf.reduce_sum(tf.matmul(inhs, inhs, transpose_b=True), axis=-1))
        # num = tf.reduce_sum(xs * inhs, axis=-1)
        # ds = num / (denom_xs * denom_inhs)
        xs = tf.nn.l2_normalize(xs, 0)
        inhs = tf.nn.l2_normalize(inhs, 0)
        # xs = tf.nn.l2_normalize(xs, -1)
        # inhs = tf.nn.l2_normalize(inhs, -1)

        # ds = 1 - tf.losses.cosine_distance(
        #     xs,
        #     inhs,
        #     axis=-1,
        #     reduction=tf.losses.Reduction.NONE)
        ds = tf.reduce_sum(xs * inhs, axis=-1)
        ds = tf.reshape(ds, [bs - 1, -1])
        # ds = tf.nn.softmax(ds, 0)
        ds_list += [ds]
    ds_list = tf.transpose(tf.concat(ds_list, -1))
    return ds_list, {}

