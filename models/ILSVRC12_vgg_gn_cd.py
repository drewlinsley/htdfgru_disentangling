#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import normalization
from layers.feedforward import conv
from layers.recurrent import recurrent_vgg16_gn_cat as vgg16
from ops import tf_fun


# def bsds_weight_decay():
#     return 0.0002


def get_aux():
    """Auxilary options for GN."""
    return {
        'attention': 'gala',  # 'gala',  # 'gala', 'se', False
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
    compression = ['pool', 'pool', 'upsample', 'upsample']
    ff_kernels = [[False]] * len(compression)
    ff_repeats = [[False]] * len(compression)
    features = [128, 256, 256, 128]  # Default
    fgru_kernels = [[3, 3], [3, 3], [1, 1], [1, 1]]
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
    data_tensor = tf.concat([data_tensor, data_tensor, data_tensor], -1)

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
            timesteps=3,
            fgru_normalization_type=normalization_type,
            ff_normalization_type=normalization_type)
        vgg(rgb=data_tensor, constructor=gammanet_constructor)
        # activity = vgg.fgru_0
    layers = 4
    ds_list = []
    bs = data_tensor.get_shape().as_list()[0]
    for idx in range(layers):
        inhs = getattr(vgg, 'inh_%s' % idx)
        xs = getattr(vgg, 'x_%s' % idx)
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
        xs = tf.nn.l2_normalize(xs, -1)
        inhs = tf.nn.l2_normalize(inhs, -1)
        ds = 1 - tf.losses.cosine_distance(
            xs,
            inhs,
            axis=-1,
            reduction=tf.losses.Reduction.NONE)
        ds = tf.reshape(ds, [bs - 1, -1])
        # ds = tf.nn.softmax(ds, 0)
        ds_list += [ds]
    ds_list = tf.transpose(tf.concat(ds_list, -1))
    return ds_list, {}
