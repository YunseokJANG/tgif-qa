"""Module for constructing RNN Cells. -- Mostly taken from tensorflow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math, numpy as np, itertools

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell
from . import highway_network_modern
from . import linear_modern as linear


RNNCell = rnn_cell.RNNCell


'''the classes here contain integrative multiplication versions of the RNN which converge faster and lead to better scores
http://arxiv.org/pdf/1606.06630v1.pdf
'''

def multiplicative_integration(list_of_inputs, output_size, initial_bias_value = 0.0,
  weights_already_calculated = False, use_highway_gate = False, use_l2_loss = False, scope = None,
  timestep = 0):
    '''expects len(2) for list of inputs and will perform integrative multiplication

    weights_already_calculated will treat the list of inputs as Wx and Uz and is useful for batch normed inputs
    '''
    with tf.variable_scope(scope or 'double_inputs_multiple_integration'):
      if len(list_of_inputs) != 2: raise ValueError('list of inputs must be 2, you have:', len(list_of_inputs))

      if weights_already_calculated: #if you already have weights you want to insert from batch norm
        Wx = list_of_inputs[0]
        Uz = list_of_inputs[1]

      else:
        with tf.variable_scope('Calculate_Wx_mulint'):
          Wx = linear.linear(list_of_inputs[0], output_size, False, use_l2_loss = use_l2_loss, timestep = timestep)
        with tf.variable_scope("Calculate_Uz_mulint"):
          Uz = linear.linear(list_of_inputs[1], output_size, False, use_l2_loss = use_l2_loss, timestep = timestep)

      with tf.variable_scope("multiplicative_integration"):
        alpha = tf.get_variable('mulint_alpha', [output_size],
            initializer = tf.truncated_normal_initializer(mean = 1.0, stddev = 0.1))

        beta1, beta2 = tf.split(0,2,
          tf.get_variable('mulint_params_betas', [output_size*2],
            initializer = tf.truncated_normal_initializer(mean = 0.5, stddev = 0.1)))

        original_bias = tf.get_variable('mulint_original_bias', [output_size],
            initializer = tf.truncated_normal_initializer(mean = initial_bias_value, stddev = 0.1))

      final_output = alpha*Wx*Uz + beta1*Uz + beta2*Wx + original_bias

      if use_highway_gate: final_output = highway_network.apply_highway_gate(final_output, list_of_inputs[0])
    return final_output


