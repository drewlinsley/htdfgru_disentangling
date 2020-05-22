#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import normalization
from layers.feedforward import conv
from layers.recurrent import recurrent_vgg16_cheap_deeper as vgg16
from ops import tf_fun


# def l1():
#     return 1e-4


def get_aux():
    """Auxilary options for GN."""
    return {
        'attention': 'gala',  # 'gala',  # 'gala',  # 'gala', 'se', False
        'attention_layers': 1,
        'norm_attention': False,
        'saliency_filter': 3,
        # 'gate_nl': tf.keras.activations.hard_sigmoid,
        'use_homunculus': False,
        'gate_homunculus': False,
        'single_homunculus': False,
        'combine_fgru_output': False,
        'upsample_nl': False,
        'upsample_convs': False,
        'separable_upsample': False,
        'separable_convs': False,  # Multiplier
        # 'fgru_output_normalization': True,
        'fgru_output_normalization': False,
        'fgru_batchnorm': True,
        'skip_connections': False,
        'residual': True,  # intermediate resid connections
        'while_loop': False,
        'image_resize': tf.image.resize_bilinear,  # tf.image.resize_nearest_neighbor
        'bilinear_init': False,
        'nonnegative': True,
        'adaptation': False,
        'symmetric_weights': 'channel',  # 'spatial_channel', 'channel', False
        'force_alpha_divisive': False,
        'force_omega_nonnegative': False,
        'td_cell_state': False,
        'td_gate': False,  # Add top-down activity to the in-gate
        'dilations': [1, 1, 1, 1],
        'partial_padding': False
    }


def v2_small():
    compression = ['pool', 'pool', 'upsample']
    ff_kernels = [[False]] * len(compression)
    ff_repeats = [[False]] * len(compression)
    features = [128, 256, 128]  # Default
    fgru_kernels = [[1, 1], [1, 1], [1, 1]]
    ar = ['']  # , 'fgru_3', 'fgru_4']  # Output layer ids
    return compression, ff_kernels, ff_repeats, features, fgru_kernels, ar


def v2_big_working():
    compression = ['pool', 'pool', 'pool', 'pool', 'upsample', 'upsample', 'upsample']
    ff_kernels = [[False]] * len(compression)
    ff_repeats = [[False]] * len(compression)
    features = [128, 256, 512, 512, 512, 256, 128]  # Default
    # features = [24, 18, 18, 48, 64, 48, 18, 18, 24]  # Bottleneck
    # features = [16, 18, 20, 48, 20, 18, 16]  # DEFAULT
    fgru_kernels = [[11, 11], [7, 7], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1]]
    fgru_kernels = [[3, 3], [3, 3], [3, 3], [3, 3], [1, 1], [1, 1], [1, 1]]
    ar = ['']  # , 'fgru_3', 'fgru_4']  # Output layer ids
    return compression, ff_kernels, ff_repeats, features, fgru_kernels, ar


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
    # norm_moments_training = training  # Force instance norm
    # normalization_type = 'no_param_batch_norm_original'
    normalization_type = 'no_param_instance_norm'
    # output_normalization_type = 'batch_norm_original_renorm'
    output_normalization_type = 'instance_norm'
    data_tensor, long_data_format = tf_fun.interpret_data_format(
        data_tensor=data_tensor,
        data_format=data_format)

    # Prepare gammanet structure
    (
        compression,
        ff_kernels,
        ff_repeats,
        features,
        fgru_kernels,
        additional_readouts) = v2_big_working()
    gammanet_constructor = tf_fun.get_gammanet_constructor(
        compression=compression,
        ff_kernels=ff_kernels,
        ff_repeats=ff_repeats,
        features=features,
        fgru_kernels=fgru_kernels)
    aux = get_aux()

    # Build model
    with tf.variable_scope('vgg', reuse=reuse):
        aux = get_aux()
        vgg = vgg16.Vgg16(
            vgg16_npy_path='/media/data_cifs/clicktionary/pretrained_weights/vgg16.npy',
            reuse=reuse,
            aux=aux,
            train=training,
            timesteps=8,
            fgru_normalization_type=normalization_type,
            ff_normalization_type=normalization_type)
        vgg(rgb=data_tensor, constructor=gammanet_constructor)
        # activity = vgg.fgru_0

    with tf.variable_scope('fgru', reuse=reuse):
        # Get side weights
        hs_0, hs_1 = [], []
        h2_rem = [
            vgg.fgru_0,
            vgg.fgru_1,
            vgg.fgru_2,
            vgg.fgru_3]
        for idx, h in enumerate(h2_rem):
            res = aux['image_resize'](
                h,
                data_tensor.get_shape().as_list()[1:3],
                align_corners=True)
            res = normalization.apply_normalization(
                activity=res,
                name='output_norm1_%s' % idx,
                normalization_type=output_normalization_type,
                data_format=data_format,
                training=training,
                trainable=training,
                reuse=reuse)
            cnv_0 = tf.layers.conv2d(
                inputs=res,
                filters=output_shape,
                kernel_size=(1, 1),
                padding='same',
                data_format=long_data_format,
                name='readout_aux_00_%s' % idx,
                # activation=tf.nn.relu,
                activation=None,
                trainable=training,
                use_bias=True,
                reuse=reuse)
            cnv_1 = tf.layers.conv2d(
                inputs=res,
                filters=output_shape,
                kernel_size=(1, 1),
                padding='same',
                data_format=long_data_format,
                name='readout_aux_10_%s' % idx,
                # activation=tf.nn.relu,
                activation=None,
                trainable=training,
                use_bias=True,
                reuse=reuse)

            hs_0 += [cnv_0]
            hs_1 += [cnv_1]

        s1, s2, s3, s4 = hs_0
        s11, s21, s31, s41 = hs_1
        o1 = tf.stop_gradient(s1)
        o2 = tf.stop_gradient(s2)
        o3 = tf.stop_gradient(s3)
        o21 = tf.stop_gradient(s21)
        o31 = tf.stop_gradient(s31)
        o41 = tf.stop_gradient(s41)

        p1_1 = s1
        p2_1 = s2 + o1
        p3_1 = s3 + o2 + o1
        p4_1 = s4 + o3 + o2 + o1
        p1_2 = s11 + o21 + o31 + o41
        p2_2 = s21 + o31 + o41
        p3_2 = s31 + o41
        p4_2 = s41

        hs = [p1_1, p2_1, p3_1, p4_1, p1_2, p2_2, p3_2, p4_2]

        activity = tf.layers.conv2d(
            tf.concat(hs, -1),
            filters=output_shape,
            kernel_size=(1, 1),
            padding='same',
            data_format=long_data_format,
            name='out',
            # activation=tf.nn.relu,
            activation=None,
            trainable=training,
            use_bias=False,
            reuse=reuse)

    if long_data_format is 'channels_first':
        activity = tf.transpose(activity, (0, 2, 3, 1))
    extra_activities = {idx: v for idx, v in enumerate(hs)}
    if activity.dtype != tf.float32:
        activity = tf.cast(activity, tf.float32)
    # return [activity, h_deep], extra_activities
    return activity, extra_activities
