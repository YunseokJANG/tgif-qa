"""Module for constructing RNN Cells with multiplicative_integration"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell

from .linear_modern import linear
from models.rnn_cell.layers import layer_norm

RNNCell = rnn_cell.RNNCell





class GRUCell_LayerNorm(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units):
    self._num_units = num_units


  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with tf.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        concated_r_u = layer_norm(linear([inputs, state], 2 * self._num_units, False, 1.0), num_variables_in_tensor = 2, initial_bias_value = 1.0)

        r, u = tf.split(1, 2, tf.sigmoid(concated_r_u))

      with tf.variable_scope("Candidate"):
        with tf.variable_scope("reset_portion"):
          reset_portion = r*layer_norm(linear([state], self._num_units, False))
        with tf.variable_scope("inputs_portion"):
          inputs_portion = layer_norm(linear([inputs], self._num_units, False))
        c = tf.tanh(reset_portion + inputs_portion)

      new_h = u * state + (1 - u) * c
    return new_h, new_h


class BasicLSTMCell_LayerNorm(RNNCell):
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
    return 2 * self._num_units

  def __call__(self, inputs, state, timestep = 0, scope=None):
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      """Long short-term memory cell (LSTM)."""
      with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
        # Parameters of gates are concatenated into one multiply for efficiency.
        h, c = tf.split(1, 2, state)

        concat = linear([inputs, h], self._num_units * 4, False, 0.0)

        concat = layer_norm(concat, reuse=None)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(1, 4, concat)

        if self.use_recurrent_dropout and self.is_training:
          input_contribution = tf.nn.dropout(tf.tanh(j), self.recurrent_dropout_factor)
        else:
          input_contribution = tf.tanh(j)

        new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * input_contribution
        with tf.variable_scope('new_h_output'):
          new_h = tf.tanh(layer_norm(new_c, reuse=None)) * tf.sigmoid(o)

      return new_h, tf.concat(1, [new_h, new_c]) #purposely reversed


class HighwayRNNCell_LayerNorm(RNNCell):
  """Highway RNN Network with multiplicative_integration"""

  def __init__(self, num_units, num_highway_layers = 3, use_inputs_on_each_layer = False):
    self._num_units = num_units
    self.num_highway_layers = num_highway_layers
    self.use_inputs_on_each_layer = use_inputs_on_each_layer


  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, timestep = 0, scope=None):
    current_state = state
    for highway_layer in xrange(self.num_highway_layers):
      with tf.variable_scope('highway_factor_'+str(highway_layer)):
        if self.use_inputs_on_each_layer or highway_layer == 0:
          highway_factor = tf.tanh(layer_norm(linear([inputs, current_state], self._num_units, False)))
        else:
          highway_factor = tf.tanh(layer_norm(linear([current_state], self._num_units, False)))
      with tf.variable_scope('gate_for_highway_factor_'+str(highway_layer)):
        if self.use_inputs_on_each_layer or highway_layer == 0:
          gate_for_highway_factor = tf.sigmoid(linear([inputs, current_state], self._num_units, True, -3.0))
        else:
          gate_for_highway_factor = tf.sigmoid(linear([current_state], self._num_units, True, -3.0))

        gate_for_hidden_factor = 1.0 - gate_for_highway_factor

      current_state = highway_factor * gate_for_highway_factor + current_state * gate_for_hidden_factor

    return current_state, current_state
