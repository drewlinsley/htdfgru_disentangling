#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    elif isinstance(output_shape, dict):
        output_shape = output_shape['output']
    data_format = 'channels_last'
    filters = 32
    depth = 9
    with tf.variable_scope('cnn', reuse=reuse):
        # Unclear if we should include l0 in the down/upsample cascade
        with tf.variable_scope('in_embedding', reuse=reuse):
            in_emb = tf.layers.conv2d(
                inputs=data_tensor,
                filters=filters,
                kernel_size=(3, 3),
                name='l0_a',
                strides=(1, 1),
                padding='same',
                activation=tf.nn.relu,
                data_format=data_format,
                trainable=training,
                use_bias=True)
            in_emb = tf.layers.conv2d(
                inputs=in_emb,
                filters=filters,
                kernel_size=(3, 3),
                name='l0_b',
                strides=(1, 1),
                padding='same',
                activation=None,
                data_format=data_format,
                trainable=training,
                use_bias=True)
        with tf.variable_scope('tower', reuse=reuse):
            for i in range(1, depth):
                branch = tf.identity(in_emb)
                in_emb = tf.nn.relu(in_emb)
                in_emb = tf.layers.conv2d(
                    inputs=in_emb,
                    filters=filters,
                    kernel_size=(3, 3),
                    name='l%s_a' % i,
                    strides=(1, 1),
                    padding='same',
                    activation=tf.nn.relu,
                    data_format=data_format,
                    trainable=training,
                    use_bias=True)
                in_emb = tf.layers.conv2d(
                    inputs=in_emb,
                    filters=filters,
                    kernel_size=(3, 3),
                    name='l%s_b' % i,
                    strides=(1, 1),
                    padding='same',
                    activation=None,
                    data_format=data_format,
                    trainable=training,
                    use_bias=True)
                in_emb += branch
            in_emb = tf.nn.relu(in_emb)

        with tf.variable_scope('readout_1', reuse=reuse):
            activity = conv.conv_layer(
                bottom=in_emb,
                name='pre_readout_conv',
                num_filters=output_shape,
                kernel_size=1,
                trainable=training,
                use_bias=True)
    extra_activities = {}
    return activity, extra_activities
