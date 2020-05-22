#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import normalization
from layers.feedforward import vgg16
from ops import tf_fun


def build_model(
        data_tensor,
        reuse,
        training,
        output_shape,
        learned_temp=False,
        data_format='NHWC'):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[-1]
    elif isinstance(output_shape, dict):
        output_shape = output_shape['output']
    # norm_moments_training = training  # Force instance norm
    # normalization_type = 'no_param_batch_norm_original'
    # output_normalization_type = 'batch_norm_original_renorm'
    output_normalization_type = 'instance_norm'
    data_tensor, long_data_format = tf_fun.interpret_data_format(
        data_tensor=data_tensor,
        data_format=data_format)

    # Build model
    with tf.variable_scope('vgg', reuse=reuse):
        vgg = vgg16.Vgg16(
            vgg16_npy_path='/media/data_cifs/clicktionary/pretrained_weights/vgg16.npy')
        vgg(rgb=data_tensor, train=training, ff_reuse=reuse)

    with tf.variable_scope('fgru', reuse=reuse):
        # Get side weights
        h2_rem = [
            vgg.conv1_2,
            vgg.conv2_2,
            vgg.conv3_3,
            vgg.conv4_3,
            vgg.conv5_3]
        res_act = []
        for idx, h in enumerate(h2_rem):
            res = normalization.apply_normalization(
                activity=h,
                name='output_norm1_%s' % idx,
                normalization_type=output_normalization_type,
                data_format=data_format,
                training=training,
                trainable=training,
                reuse=reuse)
            res_act += [tf.image.resize_bilinear(
                res,
                data_tensor.get_shape().as_list()[1:3],
                align_corners=True)]

        # # 3-step readout
        # (1) Per-pixel competition
        activity = tf.layers.conv2d(
            tf.concat(res_act, -1),
            filters=1,
            kernel_size=(3, 3),
            padding='same',
            data_format=long_data_format,
            name='out_0',
            activation=tf.nn.sigmoid,
            trainable=training,
            use_bias=True,
            reuse=reuse)

        # (2) Softmax across locations
        activity = tf.layers.conv2d(
            tf.concat(activity, -1),
            filters=1,
            kernel_size=(3, 3),
            padding='same',
            data_format=long_data_format,
            name='out_1',
            activation=None,
            trainable=training,
            use_bias=True,
            reuse=reuse)
        act_shape = activity.get_shape().as_list()
        activity = tf.reshape(activity, [act_shape[0], -1])
        if learned_temp:
            # Potentially add another non-linearity pre-GAP for VAF:q":
            temperature = tf.layers.dense(
                tf.reduce_mean(activity, reduction_indices=[1]),
                1,
                name='temperature',
                activation=None,
                use_bias=True)
            sigmoid_attention = tf.nn.sigmoid(temperature)
            map_activity = tf.nn.softmax(activity / sigmoid_attention)
        else:
            map_activity = tf.nn.softmax(activity)
        map_activity = tf.reshape(map_activity, act_shape)

        # (3) GAP & readout
        activity = tf.reduce_mean(map_activity, reduction_indices=[1, 2])
        activity = tf.layers.dense(
            activity,
            output_shape,
            name='out_2',
            reuse=reuse,
            activation=None,
            use_bias=True)

    if long_data_format is 'channels_first':
        activity = tf.transpose(activity, (0, 2, 3, 1))
    if learned_temp:
        extra_activities = {
            'temperature': temperature,
            'map_activity': map_activity
        }
    else:
        extra_activities = {
            'map_activity': map_activity
        }
    if activity.dtype != tf.float32:
        activity = tf.cast(activity, tf.float32)
    return activity, extra_activities
