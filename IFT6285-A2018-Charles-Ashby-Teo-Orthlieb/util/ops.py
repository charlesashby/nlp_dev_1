import tensorflow as tf


def MLP(input_, out_dim, size=128, scope=None):
    """ MLP Implementation """
    assert len(input_.get_shape) == 2, "MLP takes input of dimension 2 only"

    with tf.variable_scope(scope or "MLP"):
        W_h = tf.get_variable("W_hidden", [input_.get_shape()[1], size], dtype='float32')
        b_h = tf.get_variable("b_hidden", [size], dtype='float32')
        W_out = tf.get_variable("W_out", [size, out_dim], dtype='float32')
        b_out = tf.get_variable("b_out", [out_dim], dtype='float32')

    h = tf.nn.relu(tf.matmul(input_, W_h) + b_h)
    out = tf.matmul(h, W_out) + b_out
    return out


def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b
