#!/usr/bin/env python
import tensorflow as tf
from layers.recurrent import hgru_bn_for as hgru
from layers.feedforward import normalization
import numpy as np


def generate_gaussian_masks(im, sigma_list, num_centers_list=None):
    radius = im.get_shape().as_list()[0]/2
    mask = tf.zeros((radius * 2, radius * 2))
    x, y = tf.meshgrid(np.arange(-radius, radius), np.arange(-radius: radius))

    for i, sigma in enumerate(sigma_list):
        if num_centers_list is None:
            num_centers = int(40 / (float(sigma) / radius) ** 2)
        else:
            num_centers = num_centers_list[i]
        for jcenter in range(num_centers):
            xpos = tf.cast(tf.round(tf.random.uniform([], minval=-radius, maxval=radius)), tf.int32)
            ypos = tf.cast(tf.round(tf.random.uniform([], minval=-radius, maxval=radius)), tf.int32)
            bump = tf.exp(-((x-xpos) ** 2 / float(sigma) + (y-ypos) ** 2 / float(sigma)))
            mask += bump #/np.sum(bump.flatten())

    return tf.minimum(mask, 1)


def masked_additive_noise(im, mask):
    # add gaussian noise where sigma in each pixel is specified by the value of mask
    noise_img = tf.random.uniform(im.get_shape().as_list())  # np.random.normal(0.0, 1.0, im.shape)
    return tf.maximum(tf.minimum(im + noise_img * mask, 1),0)


def masked_substitution_noise(im, mask):
    # substitute an image with uniform noise where substitution probability is specified by the value of mask
    noise_img = tf.random.uniform(im.get_shape().as_list())
    substitution_img = tf.cast(tf.less(tf.random.uniform(im.get_shape().as_list()), mask), tf.int32)  # ( < mask).astype(np.int)
    return im * (1- substitution_img) + noise_img * substitution_img


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
            gauss_mask = generate_gaussian_masks(x, [20, 40], num_centers_list=[30, 15])
            o_x = masked_substitution_noise(x, gauss_mask)
            layer_hgru = hgru.hGRU(
                'hgru_1',
                x_shape=o_x.get_shape().as_list(),
                timesteps=8,
                h_ext=15,
                strides=[1, 1, 1, 1],
                padding='SAME',
                aux={'reuse': False, 'constrain': False},
                train=training)
            h2 = layer_hgru.build(o_x)

        # h2 = normalization.batch(
        #     bottom=h2,
        #     renorm=False,
        #     name='hgru_bn',
        #     training=training)

    return h2 - o_x, {}

