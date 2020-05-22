#!/usr/bin/env python
import tensorflow as tf
from layers.recurrent import hgru_bn_cd as hgru
import numpy as np


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
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
            n, h, w, c = o_x.get_shape().as_list()
            h = h - 15
            w = w - 15
            x = tf.image.resize_image_with_crop_or_pad(o_x, h, w)
            h = h - 5
            w = w - 5
            x = tf.image.random_crop(x, [n, h, w, c])
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

    layers = [[[o_x, x], inh]]
    ds_list = []
    nulls = 32
    for idx, acts in enumerate(zip(layers)):
        (x, inhs) = acts[0]
        xs = [x[1]]
        x = x[0]
        crop_size = o_x.get_shape().as_list()
        crop_size[1] = h
        crop_size[2] = w
        for idx in range(nulls):
            xs += [tf.random_crop(o_x, crop_size)]
        xs = tf.reshape(tf.concat(xs, 0), [crop_size[0] * nulls + crop_size[0], -1])
        inhs = tf.reshape(inhs, [crop_size[0], -1])
        inhs = tf.tile(inhs, [nulls + 1, 1])

        # CPC distances
        ds = tf.reduce_sum((xs * inhs), axis=-1)
        ds = tf.reshape(ds, [crop_size[0], -1])
        ds_list += [ds]
    ds_list = tf.transpose(tf.concat(ds_list, -1))
    return ds_list, {}

