#!/usr/bin/env python
import os
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
