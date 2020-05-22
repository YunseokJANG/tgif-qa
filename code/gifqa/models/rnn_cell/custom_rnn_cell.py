# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import tensorflow as tf
from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def linear(args, output_size, bias, bias_start=0.0, use_l2_loss = False, use_weight_normalization=False, scope=None, timestep = -1, weight_initializer = None, orthogonal_scale_factor = 1.1):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  # assert args #was causing error in upgraded tensorflow
  if not isinstance(args, (list, tuple)):
    args = [args]

  if len(args) > 1 and use_weight_normalization: raise ValueError('you can not use weight_normalization with multiple inputs because the euclidean norm will be incorrect -- besides, you should be using multiple integration instead!!!')

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  if use_l2_loss:
    l_regularizer = tf.contrib.layers.l2_regularizer(1e-5)
  else:
    l_regularizer = None

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size],
                      initializer = tf.uniform_unit_scaling_initializer(), regularizer = l_regularizer)
    if use_weight_normalization: matrix = weight_normalization(matrix, timestep = timestep)

    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(args, 1), matrix)

    if not bias:
      return res
    bias_term = tf.get_variable("Bias", [output_size],
                                initializer=tf.constant_initializer(bias_start), regularizer = l_regularizer)

  return res + bias_term

class LayerNormBasicLSTMCell(rnn_cell_impl.RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/pdf/1409.2329v5.pdf.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  Biases of the forget gate are initialized by default to 1 in order to reduce
  the scale of forgetting in the beginning of the training.
  """

  def __init__(self, num_units, forget_bias = 1.0, gpu_for_layer = 0, weight_initializer = "uniform_unit", orthogonal_scale_factor = 1.1, use_highway = False, num_highway_layers = 2,
    use_recurrent_dropout = False, recurrent_dropout_factor = 0.90, is_training = True):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer
    self._weight_initializer = weight_initializer
    self._orthogonal_scale_factor = orthogonal_scale_factor
    self._forget_bias = forget_bias
    self.use_highway = use_highway
    self.num_highway_layers = num_highway_layers
    self.use_recurrent_dropout = use_recurrent_dropout
    self.recurrent_dropout_factor = recurrent_dropout_factor
    self.is_training = is_training

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

  def __call__(self, inputs, state, timestep = 0, scope=None):
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      """Long short-term memory cell (LSTM)."""
      with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
        # Parameters of gates are concatenated into one multiply for efficiency.
        c, h = state

        concat = linear([inputs, h], self._num_units * 4, False, 0.0)
        concat = layers.layer_norm(concat, reuse=None)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(concat, 4, 1)

        if self.use_recurrent_dropout and self.is_training:
          input_contribution = tf.nn.dropout(tf.tanh(j), self.recurrent_dropout_factor)
        else:
          input_contribution = tf.tanh(j)

        new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * input_contribution
        with tf.variable_scope('new_h_output'):
          new_h = tf.tanh(layers.layer_norm(new_c, reuse=None)) * tf.sigmoid(o)

      new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
      # return new_h, tf.concat([new_h, new_c], 1) #purposely reversed
      return new_h, new_state


class MultiRNNCell(rnn_cell_impl.RNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, cells, state_is_tuple=True):
    """Create a RNN cell composed sequentially of a number of RNNCells.

    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  By default (False), the states are all
        concatenated along the column axis.

    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    self._cells = cells
    self._state_is_tuple = state_is_tuple
    if not state_is_tuple:
      if any(nest.is_sequence(c.state_size) for c in self._cells):
        raise ValueError("Some cells return tuples of states, but the flag "
                         "state_is_tuple is not set.  State sizes are: %s"
                         % str([c.state_size for c in self._cells]))

  @property
  def state_size(self):
    if self._state_is_tuple:
      return tuple(cell.state_size for cell in self._cells)
    else:
      return sum([cell.state_size for cell in self._cells])

  @property
  def output_size(self):
    return self._cells[-1].output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._state_is_tuple:
        return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
      else:
        # We know here that state_size of each cell is not a tuple and
        # presumably does not contain TensorArrays or anything else fancy
        return super(MultiRNNCell, self).zero_state(batch_size, dtype)

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with vs.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
      cur_state_pos = 0
      cur_inp = inputs
      new_states = []
      for i, cell in enumerate(self._cells):
        with vs.variable_scope("Cell%d" % i):
          if self._state_is_tuple:
            if not nest.is_sequence(state):
              raise ValueError(
                  "Expected state to be a tuple of length %d, but received: %s"
                  % (len(self.state_size), state))
            cur_state = state[i]
          else:
            cur_state = array_ops.slice(
                state, [0, cur_state_pos], [-1, cell.state_size])
            cur_state_pos += cell.state_size

          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)
    new_states = (tuple(new_states) if self._state_is_tuple
                  else array_ops.concat(new_states, 1))
    return cur_inp, new_states

class DropoutWrapper(rnn_cell_impl.RNNCell):
  """Operator adding dropout to inputs and outputs of the given cell."""

  def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
               seed=None):
    """Create a cell with added input and/or output dropout.

    Dropout is never used on the state.

    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      input_keep_prob: unit Tensor or float between 0 and 1, input keep
        probability; if it is float and 1, no input dropout will be added.
      output_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is float and 1, no output dropout will be added.
      seed: (optional) integer, the randomness seed.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if keep_prob is not between 0 and 1.
    """
    if not isinstance(cell, rnn_cell_impl.RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")
    if (isinstance(input_keep_prob, float) and
        not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0)):
      raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                       % input_keep_prob)
    if (isinstance(output_keep_prob, float) and
        not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0)):
      raise ValueError("Parameter output_keep_prob must be between 0 and 1: %d"
                       % output_keep_prob)
    self._cell = cell
    self._input_keep_prob = input_keep_prob
    self._output_keep_prob = output_keep_prob
    self._seed = seed

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):

    """Run the cell with the declared dropouts."""
    if (not isinstance(self._input_keep_prob, float) or
        self._input_keep_prob < 1):
      inputs = nn_ops.dropout(inputs, self._input_keep_prob, seed=self._seed)
    output, new_state = self._cell(inputs, state, scope)
    if (not isinstance(self._output_keep_prob, float) or
        self._output_keep_prob < 1):
      output = nn_ops.dropout(output, self._output_keep_prob, seed=self._seed)
    return output, new_state
