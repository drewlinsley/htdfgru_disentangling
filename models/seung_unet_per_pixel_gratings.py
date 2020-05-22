#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[-1]
    elif isinstance(output_shape, dict):
        output_shape = output_shape['output']
    data_format = 'channels_last'
    conv_kernel = [
        [3, 3],
        [3, 3],
        [3, 3],
    ]
    up_kernel = [2, 2]
    filters = [28, 36, 48, 64, 80]
    trainable = False
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
                trainable=trainable,
                use_bias=True)

        # Downsample
        l1 = conv.down_block(
            layer_name='l1',
            bottom=in_emb,
            kernel_size=conv_kernel,
            num_filters=filters[1],
            training=training,
            trainable=trainable,
            reuse=reuse)
        l2 = conv.down_block(
            layer_name='l2',
            bottom=l1,
            kernel_size=conv_kernel,
            num_filters=filters[2],
            training=training,
            trainable=trainable,
            reuse=reuse)
        l3 = conv.down_block(
            layer_name='l3',
            bottom=l2,
            kernel_size=conv_kernel,
            num_filters=filters[3],
            training=training,
            trainable=trainable,
            reuse=reuse)
        l4 = conv.down_block(
            layer_name='l4',
            bottom=l3,
            kernel_size=conv_kernel,
            num_filters=filters[4],
            training=training,
            trainable=trainable,
            reuse=reuse)

        # Upsample
        ul3 = conv.up_block(
            layer_name='ul3',
            bottom=l4,
            skip_activity=l3,
            kernel_size=up_kernel,
            num_filters=filters[3],
            training=training,
            trainable=trainable,
            reuse=reuse)
        ul3 = conv.down_block(
            layer_name='ul3_d',
            bottom=ul3,
            kernel_size=conv_kernel,
            num_filters=filters[3],
            training=training,
            trainable=trainable,
            reuse=reuse,
            include_pool=False)
        ul2 = conv.up_block(
            layer_name='ul2',
            bottom=ul3,
            skip_activity=l2,
            kernel_size=up_kernel,
            num_filters=filters[2],
            training=training,
            trainable=trainable,
            reuse=reuse)
        ul2 = conv.down_block(
            layer_name='ul2_d',
            bottom=ul2,
            kernel_size=conv_kernel,
            num_filters=filters[2],
            training=training,
            trainable=trainable,
            reuse=reuse,
            include_pool=False)
        ul1 = conv.up_block(
            layer_name='ul1',
            bottom=ul2,
            skip_activity=l1,
            kernel_size=up_kernel,
            num_filters=filters[1],
            training=training,
            trainable=trainable,
            reuse=reuse)
        ul1 = conv.down_block(
            layer_name='ul1_d',
            bottom=ul1,
            kernel_size=conv_kernel,
            num_filters=filters[1],
            training=training,
            trainable=trainable,
            reuse=reuse,
            include_pool=False)
        ul0 = conv.up_block(
            layer_name='ul0',
            bottom=ul1,
            skip_activity=in_emb,
            kernel_size=up_kernel,
            num_filters=filters[0],
            training=training,
            trainable=trainable,
            reuse=reuse)

    with tf.variable_scope('gratings_readout', reuse=reuse):
        activity = tf.layers.conv2d(
            inputs=ul0,
            filters=2,
            kernel_size=1,
            name='readout_conv_1',
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            data_format=data_format,
            trainable=True,
            use_bias=True)
        activity = tf.layers.conv2d(
            inputs=activity,
            filters=2,
            kernel_size=1,
            name='readout_conv_2',
            strides=(1, 1),
            padding='same',
            activation=None,
            data_format=data_format,
            trainable=True,
            use_bias=True)
        shapes = activity.get_shape().as_list()
        center_h = shapes[1] // 2
        center_w = shapes[2] // 2

        # Grab center column
        activity = activity[:, center_h - 20: center_h + 20, center_w - 20: center_w + 20, :]
        activity = tf.reduce_max(activity, reduction_indices=[1, 2])
    extra_activities = {}
    return activity, extra_activities
