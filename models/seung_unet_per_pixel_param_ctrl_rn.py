#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    elif isinstance(output_shape, dict):
        nhot_shape = output_shape['aux']
        output_shape = output_shape['output']
        use_aux = True
    data_format = 'channels_last'
    conv_kernel = [
        [3, 3],
        [3, 3],
        [3, 3],
    ]
    up_kernel = [2, 2]
    filters = [28, 36, 48]
    with tf.variable_scope('cnn', reuse=reuse):
        # Unclear if we should include l0 in the down/upsample cascade
        with tf.variable_scope('in_embedding', reuse=reuse):
            in_emb = tf.layers.conv2d(
                inputs=data_tensor,
                filters=filters[0],
                kernel_size=5,
                name='l0',
                strides=(1, 1),
                padding='same',
                activation=tf.nn.elu,
                data_format=data_format,
                trainable=training,
                use_bias=True)

        # Downsample
        l1 = conv.down_block(
            layer_name='l1',
            bottom=in_emb,
            kernel_size=conv_kernel,
            num_filters=filters[1],
            training=training,
            reuse=reuse)
        l2 = conv.down_block(
            layer_name='l2',
            bottom=l1,
            kernel_size=conv_kernel,
            num_filters=filters[2],
            training=training,
            include_pool=False,
            reuse=reuse)

        # Upsample
        ul0 = conv.up_block(
            layer_name='ul0',
            bottom=l2,
            skip_activity=in_emb,
            kernel_size=up_kernel,
            num_filters=filters[0],
            training=training,
            renorm=True,
            reuse=reuse)
        with tf.variable_scope('readout_1', reuse=reuse):
            activity = conv.conv_layer(
                bottom=ul0,
                name='pre_readout_conv',
                num_filters=output_shape,
                kernel_size=1,
                trainable=training,
                use_bias=True)
    extra_activities = {}
    return activity, extra_activities
