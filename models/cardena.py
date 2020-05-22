#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from layers.feedforward import normalization
from layers.feedforward import conv
from layers.recurrent import hgru_bn_for as hgru
# from layers.recurrent import hgru_bn_for_old as hgru
from ops import tf_fun


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
    output_normalization_type = 'batch_norm_original'
    ff_kernel_size = (5, 5)
    ff_nl = tf.nn.elu
    data_tensor, long_data_format = tf_fun.interpret_data_format(
        data_tensor=data_tensor,
        data_format=data_format)

    # Build model
    with tf.variable_scope('gammanet', reuse=reuse):
        conv_aux = {
            'pretrained': os.path.join(
                'weights',
                'gabors_for_contours_11.npy'),
            'pretrained_key': 's1',
            'nonlinearity': 'square'
        }
        activity = conv.conv_layer(
            bottom=data_tensor,
            name='gabor_input',
            stride=[1, 1, 1, 1],
            padding='SAME',
            trainable=training,
            use_bias=True,
            aux=conv_aux)
        layer_hgru = hgru.hGRU(
            'hgru_1',
            x_shape=activity.get_shape().as_list(),
            timesteps=8,
            h_ext=15,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux={'reuse': False, 'constrain': False},
            train=training)
        h2 = layer_hgru.build(activity)
        h2 = normalization.batch_contrib(
            bottom=h2,
            name='hgru_bn',
            training=training)
        mask = np.load('weights/cardena_mask.npy')[None, :, :, None]
        activity = h2 * mask
    with tf.variable_scope('cv_readout', reuse=reuse):
        activity = tf.reduce_mean(activity, reduction_indices=[1, 2])
        activity = tf.layers.dense(activity, output_shape)
    if long_data_format is 'channels_first':
        activity = tf.transpose(activity, (0, 2, 3, 1))
    extra_activities = {
    }
    if activity.dtype != tf.float32:
        activity = tf.cast(activity, tf.float32)
    # return [activity, h_deep], extra_activities
    return activity, extra_activities

