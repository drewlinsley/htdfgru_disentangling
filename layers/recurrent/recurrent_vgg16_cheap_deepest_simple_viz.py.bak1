import inspect
import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from ops import tf_fun
from layers.recurrent.gn_params import CreateGNParams
from layers.recurrent.gn_params import defaults
# from layers.recurrent.gammanet_refactored import GN
# from layers.recurrent.gn_recurrent_ops import GNRnOps
from layers.recurrent.gammanet_refactored_alt import GN
from layers.recurrent.gn_recurrent_ops_alt_bn import GNRnOps
from layers.recurrent.gn_feedforward_ops import GNFFOps
from layers.feedforward import normalization


def pinv(matrix):
    """Returns the Moore-Penrose pseudo-inverse"""

    s, u, v = tf.svd(matrix)
    
    threshold = tf.reduce_max(s) * 1e-5
    s_mask = tf.boolean_mask(s, s > threshold)
    s_inv = tf.diag(tf.concat([1. / s_mask, tf.zeros([tf.size(s) - tf.size(s_mask)])], 0))

    return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))


def erosion2d(img, extent):
    """Dilate a mask."""
    with tf.variable_scope('erosion2d'):
        kernel = tf.ones((extent, extent, img.get_shape()[3])) 
        output4D = tf.nn.erosion2d(img, kernel, strides=(1,1,1,1), rates=(1,1,1,1), padding="SAME")
        output4D = output4D + tf.ones_like(output4D)
        return output4D


class Vgg16(GN, CreateGNParams, GNRnOps, GNFFOps):
    def __init__(
            self,
            vgg16_npy_path,
            train,
            timesteps,
            reuse,
            fgru_normalization_type,
            ff_normalization_type,
            perturb=None,
            perturb_norm=True,
            moments_file=None,
            model_file=None,
            layer_name='recurrent_vgg16',
            ff_nl=tf.nn.relu,
            horizontal_kernel_initializer=tf.initializers.orthogonal(),
            # horizontal_kernel_initializer=tf_fun.Identity(),
            kernel_initializer=tf.initializers.orthogonal(),
            gate_initializer=tf.initializers.orthogonal(),
            train_ff_gate=None,
            train_fgru_gate=None,
            train_norm_moments=None,
            train_norm_params=None,
            train_fgru_kernels=None,
            train_fgru_params=None,
            downsampled=False, 
            up_kernel=None,
            stop_loop=False,
            recurrent_ff=False,
            perturb_method="hidden_state",  # "kernel"
            strides=[1, 1, 1, 1],
            pool_strides=[2, 2],  # Because fgrus are every other down-layer
            pool_kernel=[2, 2],
            data_format='NHWC',
            horizontal_padding='SAME',
            ff_padding='SAME',
            vgg_dtype=tf.bfloat16,
            aux=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print path
        self.perturb_norm = perturb_norm
        self.moments_file = moments_file
        self.model_file = model_file
        self.data_format = data_format
        self.pool_strides = pool_strides
        self.strides = strides
        self.pool_kernel = pool_kernel
        self.fgru_normalization_type = fgru_normalization_type
        self.ff_normalization_type = ff_normalization_type
        self.horizontal_padding = horizontal_padding
        self.ff_padding = ff_padding
        self.train = train
        self.layer_name = layer_name
        self.data_format = data_format
        self.horizontal_kernel_initializer = horizontal_kernel_initializer
        self.kernel_initializer = kernel_initializer
        self.gate_initializer = gate_initializer
        self.fgru_normalization_type = fgru_normalization_type
        self.ff_normalization_type = ff_normalization_type
        self.recurrent_ff = recurrent_ff
        self.stop_loop = stop_loop
        self.ff_nl = ff_nl
        self.fgru_connectivity = ''
        self.reuse = reuse
        self.timesteps = timesteps
        self.downsampled = downsampled
        self.last_timestep = timesteps - 1
        self.perturb = perturb
        self.perturb_method = perturb_method
        if train_ff_gate is None:
            self.train_ff_gate = self.train
        else:
            self.train_ff_gate = train_ff_gate
        if train_fgru_gate is None:
            self.train_fgru_gate = self.train
        else:
            self.train_fgru_gate = train_fgru_gate
        if train_norm_moments is None:
            self.train_norm_moments = self.train
        else:
            self.train_norm_moments = train_norm_moments
        if train_norm_moments is None:
            self.train_norm_params = self.train
        else:
            self.train_norm_params = train_norm_params
        if train_fgru_kernels is None:
            self.train_fgru_kernels = self.train
        else:
            self.train_fgru_kernels = train_fgru_kernels
        if train_fgru_kernels is None:
            self.train_fgru_params = self.train
        else:
            self.train_fgru_params = train_fgru_params

        default_vars = defaults()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                default_vars[k] = v
        self.update_params(default_vars)
        # Store variables in the order they were created. Hack for python 2.x.
        self.variable_list = OrderedDict()
        self.hidden_dict = OrderedDict()

        # Kernel info
        if data_format is 'NHWC':
            self.prepared_pool_kernel = [1] + self.pool_kernel + [1]
            self.prepared_pool_stride = [1] + self.pool_strides + [1]
            self.up_strides = [1] + self.pool_strides + [1]
        else:
            raise NotImplementedError
            self.prepared_pool_kernel = [1, 1] + self.pool_kernel
            self.prepared_pool_stride = [1, 1] + self.pool_strides
            self.up_strides = [1, 1] + self.pool_strides
        self.sanity_check()
        if self.symmetric_weights:
            self.symmetric_weights = self.symmetric_weights.split('_')

        # Nonlinearities and initializations
        if isinstance(self.recurrent_nl, basestring):
            self.recurrent_nl = tf_fun.interpret_nl(self.recurrent_nl)

        # Set initializers for greek letters
        if self.force_alpha_divisive:
            self.alpha_initializer = tf.initializers.variance_scaling
        else:
            self.alpha_initializer = tf.constant_initializer(0.1)
        self.mu_initializer = tf.constant_initializer(0.)
        self.omega_initializer = tf.constant_initializer(0.1)
        self.kappa_initializer = tf.constant_initializer(0.)

        # Handle BN scope reuse
        self.scope_reuse = reuse

        # Load weights
        self.data_dict = np.load(vgg16_npy_path, allow_pickle=True, encoding='latin1').item()
        print("npy file loaded")

    def __call__(self, rgb, label, constructor=None, store_timesteps=False):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        self.gammanet_constructor = constructor
        X_shape = rgb.get_shape().as_list()
        self.N = X_shape[0]
        self.dtype = rgb.dtype
        self.input = rgb
        self.ff_reuse = self.scope_reuse
        self.conv1_1 = self.conv_layer(self.input, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        X_shape = self.conv2_2.get_shape().as_list()
        self.prepare_tensors(X_shape, allow_resize=False)
        self.create_hidden_states(
            constructor=self.gammanet_constructor,
            shapes=self.layer_shapes,
            recurrent_ff=self.recurrent_ff,
            init=self.hidden_init,
            dtype=self.dtype)
        # self.fgru_0 = tf.get_variable(name="perturb_viz", initializer=self.conv2_2, trainable=True)
        if self.perturb is not None:
            # Load weights for deriving tuning curves
            # moments_file = "../undo_bias/neural_models/linear_moments/INSILICO_BSDS_vgg_gratings_simple_tb_feature_matrix.npz"
            # model_file = "../undo_bias/neural_models/linear_models/INSILICO_BSDS_vgg_gratings_simple_tb_model.joblib.npy"
            moments = np.load(self.moments_file)
            means = moments["means"]
            stds = moments["stds"]
            clf = np.load(self.model_file).astype(np.float32)
            clf_sq = clf.T.dot(clf)
            inv_clf = np.linalg.pinv(clf_sq).astype(np.float32)  # Precompute inversion

            # Get target units
            bs, h, w, _ = self.fgru_0.get_shape().as_list()
            hh, hw = h // 2, w // 2
            # sel_units_raw = self.fgru_0[:, hh - 2: hh + 2, hw - 2: hw + 2, :]
            # sel_units = tf.reshape(sel_units_raw, [bs, -1])  # Squeeze to a matrix

            # # Normalize activities
            # sel_units = (sel_units - means) / stds

            # # Transform to population tuning curves
            # tc = tf.matmul(tf.matmul(inv_clf, clf, transpose_b=True), sel_units, transpose_b=True)

            # Reweight tuning curves
            # tc = tc * self.perturb
            if 0:  # self.perturb != 1:  # self.perturb <= 1:
                sel_units_raw = self.fgru_0[:, hh - 2: hh + 2, hw - 2: hw + 2, :]
                sel_units = tf.reshape(sel_units_raw, [bs, -1])  # Squeeze to a matrix

                # Normalize activities
                if self.perturb_norm:
                    sel_units = (sel_units - means) / stds

                # # Transform to population tuning curves
                tc = tf.matmul(tf.matmul(inv_clf, clf, transpose_b=True), sel_units, transpose_b=True)

                # # Reweight tuning curves
                # tc = tc + self.perturb
                ###
                # tc = tf.transpose(label * self.perturb)
                if self.perturb == 1:  # ROLL RIGHT
                    tc = tf.roll(tc, 1, 0)  # Need to do all rolls [30, 60, 90] eventually
                elif self.perturb == 2:  # ROLL RIGHT
                    tc = tf.roll(tc, 2, 0)  # Need to do all rolls [30, 60, 90] eventually
                elif self.perturb == 3:  # ROLL RIGHT
                    tc = tf.roll(tc, 3, 0)  # Need to do all rolls [30, 60, 90] eventually
                elif self.perturb == -1:
                    tc = tf.roll(tc, -1, 0)  # Need to do all rolls [30, 60, 90] eventually
                elif self.perturb == -2:
                    tc = tf.roll(tc, -2, 0)  # Need to do all rolls [30, 60, 90] eventually
                elif self.perturb == -3:
                    tc = tf.roll(tc, -3, 0)  # Need to do all rolls [30, 60, 90] eventually
                else:
                    # Reweight tuning curves
                    tc = tc * self.perturb + self.perturb

            elif 0:  # self.perturb == 1:  # ROLL
                tc_inv = tf.reshape(label, [-1])  # * 200  # wtf???
            else:
                # Triganometric perturbations
                sel_units_raw = self.fgru_0[:, hh - 2: hh + 2, hw - 2: hw + 2, :]
                sel_units = tf.reshape(sel_units_raw, [bs, -1])  # Squeeze to a matrix

                # # Normalize activities
                # sel_units = (sel_units - means) / stds

                # # Transform to population tuning curves
                tc = tf.matmul(tf.matmul(inv_clf, clf, transpose_b=True), sel_units, transpose_b=True)
                a = 3.
                bin_size = 30.
                mu = tf.cast(tf.argmax(tf.squeeze(tc)), tf.float32) * bin_size  # PASS THE TUNING CENTER
                # b = -(mu - 120)   # * tf.cast(tf.less(mu, 120), tf.float32)
                # b = -60  # 90 for 150 60 for 120 30 for 90 0 for 60 -30 for 30 -60 for 0
                b = mu - 60
                # -60 is 120, (mu - 120) is -30 and +
                # orientations = np.arange(180).astype(np.float32)
                orientations = np.linspace(0, 180, 30)
                perturbation = tf.cos((tf.constant(np.pi) * (orientations - (b + mu)) / 180))
                # perturbation = tf.nn.relu(perturbation)
                perturbation = self.perturb * (perturbation ** a)

                """
                sel_units_raw = self.fgru_0[:, hh - 2: hh + 2, hw - 2: hw + 2, :]
                sel_units = tf.reshape(sel_units_raw, [bs, -1])  # Squeeze to a matrix

                # # Normalize activities
                sel_units = (sel_units - means) / stds

                # # Transform to population tuning curves
                tc = tf.matmul(tf.matmul(inv_clf, clf, transpose_b=True), sel_units, transpose_b=True)
                """

                # bin here
                binned_perturbation = tf.reduce_mean(tf.reshape(perturbation, [int(180 / bin_size), -1]), -1)
                B = tf.expand_dims(binned_perturbation, 1)
                A = tf.reshape(label, [-1, 1])
                Z = tf.matmul(A, A, transpose_b=True)
                M = tf.matmul(tf.matmul(B, A, transpose_b=True), pinv(Z))
                tc = tf.matmul(M, label, transpose_b=True)  # This is out perturbed activity
                # tc = tf.transpose(tc)
            """
            # Get gradient of tuning curves wrt target units
            tc_grad = tf.gradients(tc, sel_units)[0]
            tc_grad = tf.reshape(tc_grad, sel_units_raw.get_shape().as_list())
            """
            # Invert the inverted model
            # tc_inv = tf.matmul(tf.matmul(inv_clf, clf, transpose_b=True), tc, transpose_a=True)  # Numerical issues
            if 1:  # self.perturb != 1:
                inv_inv = np.linalg.pinv(clf.dot(clf.T))
                tc_inv = tf.matmul(tf.matmul(tf.matmul(tf.matmul(tc, clf.T, transpose_a=True), clf), clf.T), inv_inv) # predictions.T @ clf.T @ clf @ clf.T @ np.linalg.pinv(clf @ clf.T)
                tc_inv = tf.reshape(tc_inv, [-1])

                # Unnormalize activities
                if self.perturb_norm:
                    tc_inv = tc_inv * stds + means

            # Weight the target units with the gradient
            perturb_mask = np.ones(self.fgru_0.get_shape().as_list(), dtype=np.float32)
            perturb_idx = np.zeros(self.fgru_0.get_shape().as_list(), dtype=np.float32)
            perturb_bias = tf.zeros(self.fgru_0.get_shape().as_list(), dtype=tf.float32)
            self.center_h = self.fgru_0.get_shape().as_list()[1] // 2
            self.center_w = self.fgru_0.get_shape().as_list()[2] // 2
            perturb_mask[:, hh - 2: hh + 2, hw - 2: hw + 2] = 0.
            perturb_idx[:, hh - 2: hh + 2, hw - 2: hw + 2] = 1.

            # BG needs to be ignored via stopgrad
            # bg = tf.cast(tf.greater_equal(tf.reduce_mean(self.fgru_0 ** 2, reduction_indices=[-1], keep_dims=True), 10.), tf.float32)
            # self.fgru_0 = self.fgru_0 * tf.stop_gradient(bg)
            bg = tf.reduce_mean(self.conv2_2 ** 2, reduction_indices=[-1], keep_dims=True)
            bg = tf.cast(tf.greater(bg, tf.reduce_mean(bg)), tf.float32)
            bg = erosion2d(img=bg, extent=9)
            if 0:  # self.perturb < 0:
                raise RuntimeError("Negative perturbs dont make sense.")
            else:
                perturb_idxs = np.where(perturb_idx.reshape(-1))
                perturb_bias_shape = perturb_bias.get_shape().as_list()
                perturb_bias = tf.get_variable(name="perturb_bias", initializer=tf.reshape(perturb_bias, [-1]), dtype=tf.float32, trainable=False)
                perturb_bias = tf.scatter_update(perturb_bias, perturb_idxs[0], tc_inv)
                perturb_bias = tf.stop_gradient(tf.reshape(perturb_bias, perturb_bias_shape))  # Fixed perturbation
            if self.perturb_method == "kernel":
                ks = 21
                # Paramaterize the learned hidden state as conv2_2 * kernel
                self.perturb_mask = tf.get_variable(name="perturb_mask", initializer=perturb_mask, trainable=False)
                # self.perturb_bias = tf.get_variable(name="perturb_bias", initializer=perturb_bias, trainable=False)
                # fgru_kernel = tf.get_variable(name="perturb_viz", shape=(ks, ks, 128, 1), initializer=tf.compat.v1.keras.initializers.Orthogonal, trainable=True)
                fgru_kernel = tf.get_variable(name="perturb_viz", initializer=np.ones((ks, ks, 128, 1), dtype=np.float32) / (ks * ks), trainable=True)
                # if self.perturb > 1:
                #     fgru_kernel = fgru_kernel + 1  # Center around 1
                """
                fgru_kernel = tf.get_variable(name="perturb_viz", shape=(ks, ks, 1, 1), initializer=tf.compat.v1.keras.initializers.Orthogonal, trainable=True)
                sf = tf.split(self.fgru_0, self.fgru_0.get_shape().as_list()[-1], axis=-1)
                of = []
                for f in sf:
                    of.append(tf.nn.conv2d(f, fgru_kernel, strides=[1, 1, 1, 1], padding="SAME"))
                self.fgru_0 = tf.concat(of, -1)
                """
                # self.fgru_0 = tf.nn.depthwise_conv2d(tf.stop_gradient(self.conv2_2), fgru_kernel, strides=[1, 1, 1, 1], padding="SAME")
                pre_kernel = tf.stop_gradient(self.conv2_2) * perturb_mask + perturb_bias
                self.fgru_0 = tf.nn.depthwise_conv2d(pre_kernel, fgru_kernel, strides=[1, 1, 1, 1], padding="SAME")
                # self.fgru_0 = self.fgru_0 * self.perturb_mask + self.perturb_bias  # Add the bias back in!
                # self.fgru_0 = bg * (perturb_bias + (self.fgru_0 * perturb_mask)) + tf.stop_gradient(self.fgru_0 * (1 - bg))
            elif self.perturb_method == "hidden_state":
                # self.perturb_mask = tf.get_variable(name="perturb_mask", initializer=perturb_mask, trainable=False)
                mult = tf.get_variable(name="perturb_viz_mult", initializer=np.ones((self.conv2_2.get_shape().as_list())).astype(np.float32), trainable=True)  # Perturbed fgru
                add = tf.get_variable(name="perturb_viz_add", initializer=np.zeros((self.conv2_2.get_shape().as_list())).astype(np.float32), trainable=True)  # Perturbed fgru

                mult = mult * perturb_mask + tf.stop_gradient(1 - perturb_mask)
                add = add * perturb_mask + tf.stop_gradient(1 - perturb_mask)
                # self.fgru_0 = add + perturb_bias
                # self.fgru_0 = self.conv2_2 * perturb_mask + tf.stop_gradient(1 - perturb_mask)
                # self.fgru_0 = self.fgru_0 + add + perturb_bias
                self.fgru_0 = mult * self.conv2_2 + perturb_bias
                # self.fgru_0 = mult * self.fgru_0 + add + perturb_bias


                # self.fgru_0 = (add * perturb_mask + tf.stop_gradient(1 - perturb_mask)) + perturb_bias

                # self.fgru_0 = (mult * perturb_mask + add + perturb_bias) + tf.stop_gradient(1 - perturb_mask)
                # Mask the BG
                self.fgru_0 = bg * self.fgru_0 + tf.stop_gradient((1 - bg) * self.fgru_0)
                # self.fgru_0 = perturb_bias
        else:
            self.fgru_0 = self.conv2_2

        ta = []
        for idx in range(self.timesteps):
            act = self.build(i0=idx)
            self.ff_reuse = tf.AUTO_REUSE
            if store_timesteps:
                ta += [self.fgru_0]
        if self.downsampled:
            return act
        if store_timesteps:
            return ta

    def build(self, i0, extra_convs=True):
        # Convert RGB to BGR
        with tf.variable_scope('fgru'):
            error_horizontal_0, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.conv2_2,
                h2=self.fgru_0,
                layer_id=0,
                i0=i0)
        self.fgru_0 = fgru_activity  # + self.conv2_2
        self.error_0 = error_horizontal_0
        self.pool2 = self.max_pool(self.fgru_0, 'pool2')
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        if i0 == 0:
            self.fgru_1 = self.conv3_3
        with tf.variable_scope('fgru'):
            error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.conv3_3,
                h2=self.fgru_1,
                layer_id=1,
                i0=i0)
        self.error_1 = error
        self.fgru_1 = fgru_activity  # + self.conv2_2
        self.pool3 = self.max_pool(self.fgru_1, 'pool3')
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")

        if i0 == 0:
            self.fgru_2 = self.conv4_3
        with tf.variable_scope('fgru'):
            error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.conv4_3,
                h2=self.fgru_2,
                layer_id=2,
                i0=i0)
        self.fgru_2 = fgru_activity  # + self.conv3_3
        self.pool4 = self.max_pool(self.fgru_2, 'pool4')
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        if i0 == 0:
            self.fgru_3 = self.conv5_3
        with tf.variable_scope('fgru'):
            error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.conv5_3,
                h2=self.fgru_3,
                layer_id=3,
                i0=i0)
        self.fgru_3 = fgru_activity  # + self.conv5_3

        # Resize and conv
        with tf.variable_scope('fgru'):
            fgru_2_td = normalization.apply_normalization(
                activity=self.fgru_3,
                name='td_norm2_%s' % i0,
                # normalization_type='instance_norm',
                normalization_type=self.fgru_normalization_type,
                data_format=self.data_format,
                training=self.train,
                trainable=self.train,
                reuse=self.reuse)
            fgru_2_td = self.conv_layer(
                fgru_2_td,
                '4_to_3',
                learned=True,
                apply_relu=True,
                shape=[
                    1,
                    1,
                    fgru_2_td.get_shape().as_list()[-1],
                    self.fgru_2.get_shape().as_list()[-1] // 64])
            fgru_2_td = normalization.apply_normalization(
                activity=fgru_2_td,
                name='td_norm2_1%s' % i0,
                # normalization_type='instance_norm',
                normalization_type=self.fgru_normalization_type,
                data_format=self.data_format,
                training=self.train,
                trainable=self.train,
                reuse=self.reuse)
            if extra_convs:
                fgru_2_td = self.conv_layer(
                    fgru_2_td,
                    '4_to_3_2',
                    learned=True,
                    shape=[
                        1,
                        1,
                        self.fgru_2.get_shape().as_list()[-1] // 64,
                        self.fgru_2.get_shape().as_list()[-1]])
            fgru_2_td = self.image_resize(
                fgru_2_td,
                self.fgru_2.get_shape().as_list()[1:3],
                align_corners=True)
            error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.fgru_2,
                h2=fgru_2_td,
                layer_id=4,
                i0=i0)
        self.fgru_2 += fgru_activity
        if i0 == self.last_timestep and self.downsampled:
            return self.fgru_2

        # Resize and conv
        with tf.variable_scope('fgru'):
            fgru_1_td = normalization.apply_normalization(
                activity=self.fgru_2,
                name='td_norm1_%s' % i0,
                # normalization_type='instance_norm',
                normalization_type=self.fgru_normalization_type,
                data_format=self.data_format,
                training=self.train,
                trainable=self.train,
                reuse=self.reuse)
            fgru_1_td = self.conv_layer(
                fgru_1_td,
                '3_to_2',
                learned=True,
                apply_relu=True,
                shape=[
                    1,
                    1,
                    fgru_1_td.get_shape().as_list()[-1],
                    self.fgru_1.get_shape().as_list()[-1] // 16])
            fgru_1_td = normalization.apply_normalization(
                activity=fgru_1_td,
                name='td_norm1_1%s' % i0,
                # normalization_type='instance_norm',
                normalization_type=self.fgru_normalization_type,
                data_format=self.data_format,
                training=self.train,
                trainable=self.train,
                reuse=self.reuse)
            if extra_convs:
                fgru_1_td = self.conv_layer(
                    fgru_1_td,
                    '3_to_2_2',
                    learned=True,
                    shape=[
                        1,
                        1,
                        self.fgru_1.get_shape().as_list()[-1] // 16,
                        self.fgru_1.get_shape().as_list()[-1]])
            fgru_1_td = self.image_resize(
                fgru_1_td,
                self.fgru_1.get_shape().as_list()[1:3],
                align_corners=True)
            error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.fgru_1,
                h2=fgru_1_td,
                layer_id=5,
                i0=i0)
        # self.fgru_0 = fgru_activity
        self.fgru_1 += fgru_activity

        # Resize and conv
        with tf.variable_scope('fgru'):
            fgru_0_td = normalization.apply_normalization(
                activity=self.fgru_1,
                name='td_norm0_%s' % i0,
                # normalization_type='instance_norm',
                normalization_type=self.fgru_normalization_type,
                data_format=self.data_format,
                training=self.train,
                trainable=self.train,
                reuse=self.reuse)
            fgru_0_td = self.conv_layer(
                fgru_0_td,
                '2_to_1',
                learned=True,
                apply_relu=True,
                shape=[
                    1,
                    1,
                    fgru_0_td.get_shape().as_list()[-1],
                    self.fgru_0.get_shape().as_list()[-1] // 4])
            fgru_0_td = normalization.apply_normalization(
                activity=fgru_0_td,
                name='td_norm0_1%s' % i0,
                # normalization_type='instance_norm',
                normalization_type=self.fgru_normalization_type,
                data_format=self.data_format,
                training=self.train,
                trainable=self.train,
                reuse=self.reuse)
            if extra_convs:
                fgru_0_td = self.conv_layer(
                    fgru_0_td,
                    '2_to_1_2',
                    learned=True,
                    shape=[
                        1,
                        1,
                        self.fgru_0.get_shape().as_list()[-1] // 4,
                        self.fgru_0.get_shape().as_list()[-1]])
            fgru_0_td = self.image_resize(
                fgru_0_td,
                self.fgru_0.get_shape().as_list()[1:3],
                align_corners=True)
            error_td_0, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.fgru_0,
                h2=fgru_0_td,
                layer_id=6,
                i0=i0)
        self.fgru_0 = self.fgru_0 + fgru_activity
        # self.fgru_0 = fgru_activity

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(
            bottom,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name=name)

    def conv_layer(
            self,
            bottom,
            name,
            learned=False,
            shape=False,
            apply_relu=True):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name, learned=learned, shape=shape)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name, learned=learned, shape=shape)
            bias = tf.nn.bias_add(conv, conv_biases)

            if apply_relu:
                relu = tf.nn.relu(bias)
            else:
                relu = bias
            return relu

    def get_conv_filter(self, name, learned=False, shape=None):
        with tf.variable_scope('ff_vars', reuse=self.ff_reuse):
            if learned:
                return tf.get_variable(
                    name='%s_kernel' % name,
                    shape=shape,
                    dtype=self.dtype,
                    trainable=self.train,
                    initializer=tf.initializers.variance_scaling)
            else:
                return tf.get_variable(
                    name='%s_kernel' % name,
                    initializer=self.data_dict[name][0],
                    trainable=self.train)

    def get_bias(self, name, learned=False, shape=None):
        with tf.variable_scope('ff_vars', reuse=self.ff_reuse):
            if learned:
                return tf.get_variable(
                    name='%s_bias' % name,
                    shape=[shape[-1]],
                    dtype=self.dtype,
                    trainable=self.train,
                    initializer=tf.initializers.zeros)
            else:
                return tf.get_variable(
                    name='%s_bias' % name,
                    initializer=self.data_dict[name][1],
                    trainable=self.train)

