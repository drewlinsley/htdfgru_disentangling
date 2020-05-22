import tensorflow as tf


class GNRnOps(object):
    """Methods for initializing gammanet parameters."""
    def __init__(self):
        pass

    def __call__(self):
        pass

    def conv_2d_op(
            self,
            data,
            weights,
            symmetric_weights=False,
            padding=None,
            dilations=None):
        """2D convolutions for hgru."""
        if dilations is None:
            dilations = [1, 1, 1, 1]
        if padding is None:
            padding = self.horizontal_padding
        symmetric_weights = self.interpret_symmetry(symmetric_weights)
        w_shape = weights.get_shape().as_list()
        assert self.data_format == 'NHWC'
        if len(w_shape) > 1 and int(w_shape[-2]) > 1:
            # Full convolutions
            if symmetric_weights:
                g = tf.get_default_graph()
                with g.gradient_override_map({'Conv2D': symmetric_weights}):
                    activities = tf.nn.conv2d(
                        data,
                        weights,
                        self.strides,
                        padding=padding,
                        dilations=dilations,
                        data_format=self.data_format)
            else:
                activities = tf.nn.conv2d(
                    data,
                    weights,
                    self.strides,
                    padding=padding,
                    dilations=dilations,
                    data_format=self.data_format)
        else:
            raise RuntimeError
        if self.partial_padding:
            _, h, w, _ = data.get_shape().as_list()
            if h > 1 and w > 1:
                y, x, _, _ = weights.get_shape().as_list()
                _, mask_ratio = self.create_pad_mask(
                    h,
                    w,
                    y,
                    x)
                activities *= mask_ratio
        return activities

    def circuit_input(
            self,
            h2,
            var_scope,
            layer_id,
            td_gate=None):
        """Calculate gain and inh horizontal activities."""
        horizontal_kernels_inh = getattr(
            self,
            '%s_horizontal_kernels_inh_%s' % (self.symm_k_tag, layer_id))
        gain_bias = getattr(self, 'gain_bias_%s' % layer_id)

        # Prepare the input gate data
        gate_activity = h2
        if td_gate is not None and len(td_gate.get_shape()) > 1:
            if self.data_format == 'NHWC':
                gate_activity = tf.concat([gate_activity, td_gate], axis=-1)
            elif self.data_format == 'NCHW':
                gate_activity = tf.concat([gate_activity, td_gate], axis=1)
            else:
                raise RuntimeError(self.data_format)

        # Compute input gate with attention or not
        if self.attention is 'se' or self.attention is 'gala':
            g1_intermediate = self.apply_attention(
                activity=tf.reduce_max(
                    gate_activity,
                    reduction_indices=[1, 2],
                    keep_dims=True),
                layer_id=layer_id,
                var_scope=var_scope,
                dilations=self.dilations,
                attention='global')

        if self.attention is 'gala':
            g1_spatial = self.apply_attention(
                activity=gate_activity,
                layer_id=layer_id,
                var_scope=var_scope,
                dilations=self.dilations,
                attention='spatial')

            # # Force multiplicative interactions
            # g1_spatial_shape = g1_spatial.get_shape().as_list()
            # g1_spatial = tf.reshape(
            #     g1_spatial,
            #     [
            #         g1_spatial_shape[0],
            #         g1_spatial_shape[1] * g1_spatial_shape[2]])
            # g1_spatial = tf.nn.softmax(g1_spatial, axis=1)
            # g1_spatial = tf.reshape(g1_spatial, g1_spatial_shape)

            # Force multiplicative interactions
            g1_intermediate = g1_intermediate * g1_spatial

        if not self.attention:
            gain_kernels = getattr(self, '%s_gain_kernels_%s' % (
                self.symm_g_tag, layer_id))
            g1_intermediate = self.conv_2d_op(
                data=gate_activity,
                weights=gain_kernels,
                symmetric_weights=self.symmetric_gate_weights,
                dilations=self.dilations)

        # Batchnorm the gate activity before applying a sigmoid
        if self.fgru_batchnorm:
            with tf.variable_scope(
                    '%s/g1_bn' % var_scope,
                    reuse=self.scope_reuse) as scope:
                beta = getattr(self, 'g1_bn_beta_%s' % layer_id)
                gamma = getattr(self, 'g1_bn_gamma_%s' % layer_id)
                g1_intermediate = self.nl_bn(
                    activity=g1_intermediate,
                    ff_name='%s/g1_bn' % var_scope,
                    nl=None,
                    normalization_type=self.fgru_normalization_type,
                    scope=scope,
                    gamma=gamma,
                    beta=beta)
        h2 = h2 * self.gate_nl(g1_intermediate + gain_bias)

        # Horizontal activities
        c1 = self.conv_2d_op(
            data=h2,
            weights=horizontal_kernels_inh,
            symmetric_weights=self.symmetric_weights,
            dilations=self.dilations)
        if self.fgru_batchnorm and self.c1_c2_norm:
            with tf.variable_scope(
                    '%s/c1_bn' % var_scope,
                    reuse=self.scope_reuse) as scope:
                beta = getattr(self, 'c1_bn_beta_%s' % layer_id)
                gamma = getattr(self, 'c1_bn_gamma_%s' % layer_id)
                c1 = self.nl_bn(
                    activity=c1,
                    ff_name='%s/c1_bn' % var_scope,
                    nl=None,
                    normalization_type=self.fgru_normalization_type,
                    scope=scope,
                    gamma=gamma,
                    beta=beta)
        return c1

    def circuit_output(self, h1, var_scope, layer_id):
        """Calculate mix and exc horizontal activities."""
        mix_kernels = getattr(self, '%s_mix_kernels_%s' % (
            self.symm_g_tag, layer_id))
        mix_bias = getattr(self, 'mix_bias_%s' % layer_id)
        horizontal_kernels_exc = getattr(
            self, '%s_horizontal_kernels_exc_%s' % (self.symm_k_tag, layer_id))

        # Mix gate
        g2_intermediate = self.conv_2d_op(
            data=h1,
            weights=mix_kernels,
            symmetric_weights=self.symmetric_gate_weights,
            dilations=self.dilations)
        if self.fgru_batchnorm:
            with tf.variable_scope(
                    '%s/g2_bn' % var_scope,
                    reuse=self.scope_reuse) as scope:
                beta = getattr(self, 'g2_bn_beta_%s' % layer_id)
                gamma = getattr(self, 'g2_bn_gamma_%s' % layer_id)
                g2_intermediate = self.nl_bn(
                    activity=g2_intermediate,
                    ff_name='%s/g2_bn' % var_scope,
                    nl=None,
                    normalization_type=self.fgru_normalization_type,
                    scope=scope,
                    gamma=gamma,
                    beta=beta)

        g2 = self.gate_nl(g2_intermediate + mix_bias)

        # Horizontal activities
        c2 = self.conv_2d_op(
            data=h1,
            weights=horizontal_kernels_exc,
            symmetric_weights=self.symmetric_weights,
            dilations=self.dilations)
        if self.fgru_batchnorm and self.c1_c2_norm:
            with tf.variable_scope(
                    '%s/c2_bn' % var_scope,
                    reuse=self.scope_reuse) as scope:
                beta = getattr(self, 'c2_bn_beta_%s' % layer_id)
                gamma = getattr(self, 'c2_bn_gamma_%s' % layer_id)
                c2 = self.nl_bn(
                    activity=c2,
                    ff_name='%s/c2_bn' % var_scope,
                    nl=None,
                    normalization_type=self.fgru_normalization_type,
                    scope=scope,
                    gamma=gamma,
                    beta=beta)
        return c2, g2

    def input_integration(self, x, c1, h2, layer_id):
        """Integration on the input."""
        mu = getattr(self, 'mu_%s' % layer_id)
        alpha = getattr(self, 'alpha_%s' % layer_id)
        if self.force_alpha_divisive:
            alpha = tf.nn.sigmoid(alpha)

        # Nonnegative constraint on horiz interactions with x
        inh = (alpha * h2 + mu) * c1
        if self.nonnegative:
            return self.recurrent_nl(
                self.recurrent_nl(x) - self.recurrent_nl(inh))
        else:
            return self.recurrent_nl(x - inh)

    def output_integration(self, h1, c2, g2, h2, layer_id):
        """Integration on the output."""
        if self.multiplicative_excitation:
            # + and * interactions
            # kappa = getattr(self, 'kappa_%s' % layer_id)
            # omega = getattr(self, 'omega_%s' % layer_id)
            # if self.force_omega_nonnegative:
            #     omega = tf.nn.relu(omega)
            # h2_hat = self.recurrent_nl(h1 * (kappa + omega * c2))

            kappa = getattr(self, 'kappa_%s' % layer_id)
            omega = getattr(self, 'omega_%s' % layer_id)
            a = kappa * (h1 + c2)
            m = omega * (h1 * c2)
            h2_hat = self.recurrent_nl(a + m)
        else:
            # + interactions
            h2_hat = self.recurrent_nl(h1 + c2)
        return (1 - g2) * h2 + g2 * h2_hat

    def fgru_ops(
            self,
            i0,
            ff_drive,
            h2,
            layer_id,
            td_gate=None,
            td_cell=None):
        """fGRU body."""
        fgru_name = '%s_fgru_weights' % layer_id
        if not self.while_loop:
            fgru_name = '%s_t%s' % (fgru_name, i0)

        # Circuit input receives recurrent output h2
        c1 = self.circuit_input(
            h2=h2,
            var_scope=fgru_name,
            layer_id=layer_id)

        # Calculate input (-) integration: h1 (4)
        h1 = self.input_integration(
            x=ff_drive,
            c1=c1,
            h2=h2,
            layer_id=layer_id)

        # Circuit output receives recurrent input h1
        c2, g2 = self.circuit_output(
            h1=h1,
            var_scope=fgru_name,
            layer_id=layer_id)

        # Calculate output (+) integration: h2 (8, 9)
        h2 = self.output_integration(
            h1=h1,
            c2=c2,
            g2=g2,
            h2=h2,
            layer_id=layer_id)

        # Normalize if requested
        if self.adaptation is 'eta':
            eta = getattr(self, 'eta_%s' % layer_id)
            e = tf.gather(eta, i0, axis=-1)
            h2 = h2 * e
        return h1, h2

    def fgru_postprocess(
            self,
            activity,
            error,
            layer_id):
        """Routines for combining and normalizing fgru activities."""
        if self.combine_fgru_output:
            activity = tf.concat([activity, error], axis=-1)

        if self.fgru_output_normalization:
            fgru_name = '%s_fgru_weights' % layer_id
            with tf.variable_scope(
                    '%s/h2_bn' % fgru_name,
                    reuse=self.scope_reuse) as scope:
                beta = getattr(self, 'h2_bn_beta_%s' % layer_id)
                gamma = getattr(self, 'h2_bn_gamma_%s' % layer_id)
                activity = self.nl_bn(
                    activity=activity,
                    ff_name='%s/h2_bn' % fgru_name,
                    nl=None,
                    normalization_type=self.fgru_normalization_type,
                    scope=scope,
                    gamma=gamma,
                    beta=beta)
        return activity

    def apply_attention(
            self,
            activity,
            layer_id,
            dilations,
            attention,
            var_scope):
        """Compute the attention passes from GALA w/ convs."""
        assert attention is 'spatial' or attention is 'global'
        for g_idx in range(self.attention_layers):
            att_kernel_tag = '%s_%s_%s_%s_gain' % (
                self.symm_g_tag, g_idx, layer_id, attention)
            att_bias_tag = '%s_%s_%s_%s_bias' % (
                self.symm_g_tag, g_idx, layer_id, attention)
            att_kernel = getattr(self, att_kernel_tag)
            att_bias = getattr(self, att_bias_tag)
            activity = self.conv_2d_op(
                data=activity,
                weights=att_kernel,
                symmetric_weights=False,
                dilations=dilations)
            activity = self.bias_add(
                activity,
                att_bias)
            # if g_idx < (self.attention_layers - 1) and self.norm_attention:
            if self.norm_attention:
                beta = getattr(
                    self, '%s_%s_%s_beta' % (g_idx, layer_id, attention))
                gamma = getattr(
                    self, '%s_%s_%s_gamma' % (g_idx, layer_id, attention))
                activity = self.nl_bn(
                    activity,
                    ff_name='%s_%s_bn' % (var_scope, att_kernel_tag),
                    gamma=gamma,
                    beta=beta,
                    normalization_type=self.ff_normalization_type,
                    nl=self.ff_nl)
            elif g_idx < (self.attention_layers - 1):
                activity = self.ff_nl(activity)
        return activity
