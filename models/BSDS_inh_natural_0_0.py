#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import normalization
from layers.feedforward import conv
# from layers.recurrent import recurrent_vgg16_cheap_deepest_simple_viz as vgg16
from layers.recurrent import recurrent_vgg16_cheap_deepest_simple_viz_inh as vgg16
from ops import tf_fun
from ops import gradients
import numpy as np


# def bsds_weight_decay():
#     return 0.0002


def dilation2d(img, extent):
    """Dilate a mask."""
    with tf.variable_scope('dilation2d'):
        kernel = tf.ones((extent, extent, img.get_shape()[3]))
        output4D = tf.nn.erosion2d(img, kernel, strides=(1,1,1,1), rates=(1,1,1,1), padding="SAME")
        output4D = output4D + tf.ones_like(output4D)
        return output4D


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
    compression = ['pool', 'pool', 'pool', 'pool', 'upsample', 'upsample', 'upsample']
    ff_kernels = [[False]] * len(compression)
    ff_repeats = [[False]] * len(compression)
    # features = [24, 18, 18, 48, 64, 48, 18, 18, 24]  # Bottleneck
    features = [128, 256, 512, 512, 512, 256, 128]  # Default
    # features = [16, 18, 20, 48, 20, 18, 16]  # DEFAULT
    # fgru_kernels = [[11, 11], [7, 7], [5, 5], [3, 3], [1, 1], [1, 1], [1, 1]]
    fgru_kernels = [[9, 9], [5, 5], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1]]
    fgru_kernels = [[9, 9], [3, 3], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1]]
    fgru_kernels = [[3, 3], [3, 3], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1]]
    # fgru_kernels = [[3, 3], [3, 3], [3, 3], [3, 3], [1, 1], [1, 1], [1, 1]]
    ar = ['']  # , 'fgru_3', 'fgru_4']  # Output layer ids
    return compression, ff_kernels, ff_repeats, features, fgru_kernels, ar


def build_model(
        data_tensor,
        labels,
        reuse,
        training,
        output_shape,
        perturb_norm=False,
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
        hh, hw = 111, 121
        aux = get_aux()
        fb_moments_file = "../undo_bias/neural_models/linear_moments/tb_feature_matrix.npz"
        fb_model_file = "../undo_bias/neural_models/linear_models/tb_model.joblib.npy"
        ff_moments_file = "../undo_bias/neural_models/linear_moments/conv2_2_tb_feature_matrix.npz"
        ff_model_file = "../undo_bias/neural_models/linear_models/conv2_2_tb_model.joblib.npy"
        vgg = vgg16.Vgg16(
            vgg16_npy_path='/media/data_cifs_lrs/clicktionary/pretrained_weights/vgg16.npy',
            reuse=reuse,
            aux=aux,
            hh=hh,
            hw=hw,
            moments_file=ff_moments_file,
            model_file=ff_model_file,
            train=False,
            timesteps=8,
            perturb=1.1,  # 70 might work  # 2.,  # 1.001,  # 17.1,
            perturb_norm=perturb_norm,
            # perturb=1.5,  # 2.,  # 1.001,  # 17.1,
            # perturb=2.,  # 2.,  # 1.001,  # 17.1,
            fgru_normalization_type=normalization_type,
            ff_normalization_type=normalization_type)
        # gn = tf.get_default_graph()
        # with gn.gradient_override_map({'Conv2D': 'PerturbVizGrad'}):
            # Scope the entire GN (with train=False).
            # The lowest-level recurrent tensor is the only
            # trainable tensor. This grad op will freeze the
            # center unit, forcing other units to overcompensate
            # to recreate a model's prediction.
            # TODO: need to get the original model output... could precompute this.  # noqa
        vgg(rgb=data_tensor, label=labels, constructor=gammanet_constructor)
        activity = vgg.fgru_0

        # Load tuning curve transform
        moments = np.load(fb_moments_file)
        means = moments["means"]
        stds = moments["stds"]
        clf = np.load(fb_model_file).astype(np.float32)

        # Transform activity to outputs
        bs, h, w, _ = vgg.fgru_0.get_shape().as_list()
        sel_units = tf.reshape(vgg.fgru_0[:, hh - 2: hh + 2, hw - 2: hw + 2, :], [bs, -1])
        # sel_units = tf.reshape(vgg.fgru_0[:, hh: hh + 4, hw: hw + 4, :], [bs, -1])
        # grad0 = tf.gradients(sel_units, vgg.conv2_2)[0]

        if perturb_norm:
            # Normalize activities -- Not normalized!!
            sel_units = (sel_units - means) / stds

        # Map responses
        # inv_clf = np.linalg.inv(clf.T.dot(clf)).astype(np.float32)  # Precompute inversion
        # inv_clf = tf.linalg.inv(tf.matmul(clf, clf, transpose_a=True))
        inv_clf = np.linalg.inv(clf.T.dot(clf)).astype(np.float32)  # Precompute inversion
        activity = tf.matmul(
            tf.matmul(inv_clf, clf, transpose_b=True), sel_units, transpose_b=True)

    impatch = data_tensor[:, hh * 2 - 14: hh * 2 + 18, hw * 2 - 14: hw * 2 + 18]
    mx = max(h, w)
    x, y = np.meshgrid(np.arange(mx), np.arange(mx))
    x = x[:h, :w]
    y = y[:h, :w]
    xy = np.stack((x, y), -1)
    coord = np.asarray([hh, hw])[None]
    dist = np.sqrt(((xy - coord) ** 2).mean(-1))
    dist = dist / dist.max()
    # penalty = tf.reduce_mean(tf.tanh(tf.abs(tf.squeeze(vgg.fgru_0))), -1) * dist
    # dist = dist * tf.squeeze(vgg.mask)  # zero out units in the RF
    # penalty = tf.cast(tf.sigmoid(tf.reduce_mean(tf.abs(tf.squeeze(vgg.fgru_0)), -1)) - 0.5, tf.float32) * dist.astype(np.float32)
    penalty = tf.cast(tf.sigmoid(tf.reduce_mean(tf.abs(tf.squeeze(vgg.mult)), -1)) - 0.5, tf.float32) * dist.astype(np.float32)
    penalty = tf.cast(penalty, tf.float32) * tf.cast(1. - tf.squeeze(vgg.mask), tf.float32)  # 0 values in the H-diameter
    penalty = penalty * 0.0
    extra_activities = {"fgru": vgg.fgru_0, "mask": vgg.mask, "penalty": penalty, "impatch": impatch}  # tf.get_variable(name="perturb_viz")}  # idx: v for idx, v in enumerate(hs_0)}
    if activity.dtype != tf.float32:
        activity = tf.cast(activity, tf.float32)
    # return [activity, h_deep], extra_activities
    return activity, extra_activities

