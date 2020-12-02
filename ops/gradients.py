import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops


# Perturbation grads
try:
    @tf.RegisterGradient('PerturbVizGrad')
    def _PerturbConv2DGrad(op, grad):
        """Set grads for the middle-most column to 0."""
        strides = op.get_attr('strides')
        padding = op.get_attr('padding')
        use_cudnn_on_gpu = op.get_attr('use_cudnn_on_gpu')
        data_format = op.get_attr('data_format')
        shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])
        dx = nn_ops.conv2d_backprop_input(
            shape_0,
            op.inputs[1],
            grad,
            strides=strides,
            padding=padding,
            use_cudnn_on_gpu=use_cudnn_on_gpu,
            data_format=data_format)
        dw = nn_ops.conv2d_backprop_filter(
            op.inputs[0],
            shape_1,
            grad,
            strides=strides,
            padding=padding,
            use_cudnn_on_gpu=use_cudnn_on_gpu,
            data_format=data_format)

        # # Find middle unit
        # h, w = shape_1[1] // 2, shape_1[2] // 2
        # # Set middle unit gradient to 0
        # dx[:, h, w] = 0.

        # Find middle unit of hidden state (these are weights)
        h, w = shape_0[1] // 2, shape_0[2] // 2
        # Set middle unit gradient to 0
        dw[:, h, w] = 0.
        return dx, dw
except Exception, e:
    print str(e)
    print 'Already imported SymmetricConv.'

# Symmetric grads
try:
    @tf.RegisterGradient('ChannelSymmetricConv')
    def _CConv2DGrad(op, grad):
        """Weight sharing for symmetric lateral connections."""
        strides = op.get_attr('strides')
        padding = op.get_attr('padding')
        use_cudnn_on_gpu = op.get_attr('use_cudnn_on_gpu')
        data_format = op.get_attr('data_format')
        shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])
        dx = nn_ops.conv2d_backprop_input(
            shape_0,
            op.inputs[1],
            grad,
            strides=strides,
            padding=padding,
            use_cudnn_on_gpu=use_cudnn_on_gpu,
            data_format=data_format)
        dw = nn_ops.conv2d_backprop_filter(
            op.inputs[0],
            shape_1,
            grad,
            strides=strides,
            padding=padding,
            use_cudnn_on_gpu=use_cudnn_on_gpu,
            data_format=data_format)
        dw = 0.5 * (dw + tf.transpose(dw, (0, 1, 3, 2)))
        return dx, dw
except Exception, e:
    print str(e)
    print 'Already imported SymmetricConv.'


try:
    @tf.RegisterGradient('SpatialSymmetricConv')
    def _SConv2DGrad(op, grad):
        """Weight sharing for symmetric lateral connections."""
        strides = op.get_attr('strides')
        padding = op.get_attr('padding')
        use_cudnn_on_gpu = op.get_attr('use_cudnn_on_gpu')
        data_format = op.get_attr('data_format')
        shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])
        dx = nn_ops.conv2d_backprop_input(
            shape_0,
            op.inputs[1],
            grad,
            strides=strides,
            padding=padding,
            use_cudnn_on_gpu=use_cudnn_on_gpu,
            data_format=data_format)
        dw = nn_ops.conv2d_backprop_filter(
            op.inputs[0],
            shape_1,
            grad,
            strides=strides,
            padding=padding,
            use_cudnn_on_gpu=use_cudnn_on_gpu,
            data_format=data_format)
        dw = 0.5 * (dw + dw[::-1, ::-1, :, :])
        return dx, dw
except Exception, e:
    print str(e)
    print 'Already imported SymmetricConv.'


try:
    @tf.RegisterGradient('SpatialChannelSymmetricConv')
    def _SCConv2DGrad(op, grad):
        """Weight sharing for symmetric lateral connections."""
        strides = op.get_attr('strides')
        padding = op.get_attr('padding')
        use_cudnn_on_gpu = op.get_attr('use_cudnn_on_gpu')
        data_format = op.get_attr('data_format')
        shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])
        dx = nn_ops.conv2d_backprop_input(
            shape_0,
            op.inputs[1],
            grad,
            strides=strides,
            padding=padding,
            use_cudnn_on_gpu=use_cudnn_on_gpu,
            data_format=data_format)
        dw = nn_ops.conv2d_backprop_filter(
            op.inputs[0],
            shape_1,
            grad,
            strides=strides,
            padding=padding,
            use_cudnn_on_gpu=use_cudnn_on_gpu,
            data_format=data_format)
        dw = 0.5 * (dw + dw[::-1, ::-1])
        dw = 0.5 * (dw + tf.transpose(dw, (0, 1, 3, 2)))
        return dx, dw
except Exception, e:
    print str(e)
    print 'Already imported SymmetricConv.'

