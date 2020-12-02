#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import normalization
from layers.feedforward import conv
from layers.recurrent import recurrent_vgg16_cheap_deepest_simple_viz as vgg16
from ops import tf_fun
from ops import gradients
import numpy as np


# def bsds_weight_decay():
#     return 0.0002
def pinv(matrix):
    """Returns the Moore-Penrose pseudo-inverse"""

    s, u, v = tf.svd(matrix)

    threshold = tf.reduce_max(s) * 1e-5
    s_mask = tf.boolean_mask(s, s > threshold)
    s_inv = tf.diag(tf.concat([1. / s_mask, tf.zeros([tf.size(s) - tf.size(s_mask)])], 0))

    return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))


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
            vgg16_npy_path='/media/data_cifs_lrs/clicktionary/pretrained_weights/vgg16.npy',
            reuse=reuse,
            aux=aux,
            train=False,
            timesteps=8,
            perturb=1,  # 1.3,  #  + 1e-8,  # 1e-5,
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

        # Load tuning curve transform
        moments_file = "../undo_bias/neural_models/linear_moments/INSILICO_BSDS_vgg_gratings_simple_tb_feature_matrix.npz"
        model_file = "../undo_bias/neural_models/linear_models/INSILICO_BSDS_vgg_gratings_simple_tb_model.joblib.npy"
        moments = np.load(moments_file)
        means = moments["means"]
        stds = moments["stds"]
        clf = np.load(model_file).astype(np.float32)

        # Transform activity to outputs
        bs, h, w, _ = vgg.fgru_0.get_shape().as_list()
        hh, hw = h // 2, w // 2
        sel_units = tf.reshape(vgg.fgru_0[:, hh - 2: hh + 2, hw - 2: hw + 2, :], [bs, -1])

        # Normalize activities
        sel_units = (sel_units - means) / stds

        # Map responses
        # inv_clf = np.linalg.inv(clf.T.dot(clf)).astype(np.float32)  # Precompute inversion
        # inv_clf = pinv(tf.matmul(clf, clf, transpose_a=True))
        inv_clf = np.linalg.inv(clf.T.dot(clf))
        inv_matmul = inv_clf.dot(clf.T)
        # activity = tf.matmul(
        #     tf.matmul(inv_clf, clf, transpose_b=True), sel_units, transpose_b=True)
        activity = tf.matmul(inv_matmul, sel_units, transpose_b=True)

    extra_activities = {"fgru": vgg.fgru_0}  # tf.get_variable(name="perturb_viz")}  # idx: v for idx, v in enumerate(hs_0)}
    if activity.dtype != tf.float32:
        activity = tf.cast(activity, tf.float32)
    # return [activity, h_deep], extra_activities
    return activity, extra_activities

