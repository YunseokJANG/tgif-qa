import math
from contextlib import contextmanager
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

def conv2d(input_, output_size, k_h=3, k_w=3, stddev=0.02, scope="conv2d"):
    with tf.variable_scope(scope):
        w = tf.get_variable('w_2d', [k_h, k_w, input_.get_shape()[-1], output_size],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_size], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv

def conv1d(input_, output_size, k_w=32, stddev=0.02, scope="conv1d"):
    input3d = tf.expand_dims(input_, 2)
    with tf.variable_scope(scope):
        w = tf.get_variable('w_1d', [k_w, 1, 1],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv1d(input3d, w, stride=1, padding='SAME')
        conv = tf.reshape(conv, input_.get_shape())
        biases = tf.get_variable('biases', [output_size], initializer=tf.constant_initializer(0.0))
        return tf.nn.bias_add(conv, biases)

'''
def linear(input_, output_size, name="linear", activation_fn=None, reuse=False):
    res = tf.contrib.layers.fully_connected(
        input_, output_size, activation_fn=activation_fn, reuse=reuse, scope=name)
    return res
'''
def linear(input_, output_size, name="linear", activation_fn=None, reuse=True):
    with tf.variable_scope(name):

        w = tf.get_variable('linear_w', [input_.get_shape()[-1], output_size],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('linear_b', [output_size], initializer=tf.constant_initializer(0.0))
        res = tf.add(tf.matmul(input_, w), b)
        if activation_fn:
            res = activation_fn(res)

    return res

'''
input:
    video = (batch_size, seq_length, 1, 1, feat_dim)
    video_mask = (batch_size, seq_length)
output:
    video = (batch_size, feat_dim) or (batch_size*seq_length, feat_dim)
'''
def aggregate_video(video, video_mask, is_aggregate):
    feat_dim = video.get_shape().as_list()
    if is_aggregate:
        video_len = tf.reduce_sum(video_mask, 1, keep_dims=True)
        video_len = tf.reshape(video_len, [-1, 1, 1, 1])
        video_len_tiled = tf.tile(video_len, [1] + feat_dim[2:])
        video_agg = tf.div(tf.reduce_sum(video, [1]), video_len_tiled)
    else:
        video_agg = tf.reshape(video, [-1] + feat_dim[2:])
    return video_agg

'''
input:
    caption = (batch_size, seq_length, feat_dim)
    caption_mask = (batch_size, seq_length)
output:
    caption = (batch_size, seq_length, feat_dim)
            or (batch_size*seq_length, seq_length, feat_dim)
'''
def aggregate_caption(caption, caption_mask, is_aggregate):
    seq_len = caption.get_shape()[-1].value
    if is_aggregate:
        caption_agg = caption
        caption_mask_agg = caption_mask
    else:
        caption_agg = tf.reshape(tf.tile(caption, [1, seq_len]),
                                 [-1, seq_len])
        caption_mask_agg = tf.reshape(tf.tile(caption_mask, [1, seq_len]),
                                      [-1, seq_len])
    return caption_agg, caption_mask_agg

'''
input:
    answer = (batch_size, 1)
output:
    caption = (batch_size, 1) or (batch_size*seq_length, 1)
'''
def aggregate_answer(answer, seq_len, is_aggregate):
    if is_aggregate:
        answer_agg = tf.reshape(answer, [-1, 1])
    else:
        answer_agg = tf.reshape(tf.tile(answer, [1, seq_len]), [-1, 1])
    return answer_agg

'''
reduce mean of loss or accuracy
input:
    values = (batch_size) or (batch_size * seq_len)
    agg_type = "min" or "max" or "avg"
output:
    scalar
'''
def aggregate_reduce_mean(values, agg_type, seq_len, is_aggregate, name):
    values = tf.cast(values, tf.float32)
    if is_aggregate:
        values_agg = values
    else:
        values = tf.reshape(values, [-1, seq_len])
        if agg_type is "min":
            values_agg = tf.reduce_min(values, 1)
        elif agg_type is "max":
            values_agg = tf.reduce_max(values, 1)
        elif agg_type is "avg":
            values_agg = tf.reduce_mean(values, 1)
    return tf.reduce_mean(values_agg, name=name)

@contextmanager
def variables_on_cpu():
    old_fn = tf.get_variable
    def new_fn(*args, **kwargs):
        with tf.device("/cpu:0"):
            return old_fn(*args, **kwargs)
    tf.get_variable = new_fn
    yield
    tf.get_variable = old_fn

