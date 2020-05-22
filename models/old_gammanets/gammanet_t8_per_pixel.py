#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.recurrent import feedback_hgru_for_v4 as feedback_hgru


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        # Add input
        in_emb = conv.skinny_input_layer(
            X=data_tensor,
            reuse=reuse,
            training=training,
            features=24,
            conv_activation=tf.nn.elu,
            conv_kernel_size=7,
            pool=False,
            name='l0')
        layer_hgru = feedback_hgru.hGRU(
            layer_name='hgru_1',
            x_shape=in_emb.get_shape().as_list(),
            timesteps=8,
            h_ext=[[9, 9], [5, 5], [1, 1]],
            strides=[1, 1, 1, 1],
            pool_strides=[4, 4],
            padding='SAME',
            aux={
                'symmetric_weights': True,
                'dilations': [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                'batch_norm': True,
                'while_loop': False,
                'nonnegative': True,
                'residual': True,
                'final_bn': False,
                'renorm': False,
                'pooling_kernel': [4, 4],
                'intermediate_ff': [24, 24],  # + filters,
                'intermediate_ks': [[5, 5], [5, 5]]},
            train=training)
        h2 = layer_hgru.build(in_emb)
        nh2 = normalization.batch(
            bottom=h2,
            name='hgru_bn',
            fused=True,
            renorm=True,
            training=training)
        activity = conv.conv_layer(
            bottom=nh2,
            name='pre_readout_conv',
            num_filters=output_shape['output'],
            kernel_size=1,
            trainable=training,
            use_bias=True)

    extra_activities = {
        # 'activity': h2
    }
    return activity, extra_activities
