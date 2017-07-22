# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Basic linear combinations that implicitly generate variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


use_weight_normalization_default = False
def linear(args, output_size, bias, bias_start=0.0, use_l2_loss = False, use_weight_normalization = use_weight_normalization_default, scope=None, timestep = -1, weight_initializer = None, orthogonal_scale_factor = 1.1): 
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
      res = tf.matmul(tf.concat(1, args), matrix)

    if not bias:
      return res
    bias_term = tf.get_variable("Bias", [output_size],
                                initializer=tf.constant_initializer(bias_start), regularizer = l_regularizer)

  return res + bias_term


def batch_timesteps_linear(input, output_size, bias, bias_start=0.0, use_l2_loss = False, use_weight_normalization = use_weight_normalization_default, scope=None,
  tranpose_input = True, timestep = -1):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 3D Tensor [timesteps, batch_size, input_size]
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
  # Calculate the total size of arguments on dimension 2.
  if tranpose_input: 
    input = tf.transpose(input, [1,0,2])

  shape_list = input.get_shape().as_list()
  if len(shape_list) != 3: raise ValueError('shape must be of size 3, you have inputted shape size of:', len(shape_list))

  num_timesteps = shape_list[0]
  batch_size = shape_list[1]
  total_arg_size = shape_list[2] 

  if use_l2_loss:
    l_regularizer = tf.contrib.layers.l2_regularizer(1e-5)
  else:
    l_regularizer = None

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size], initializer = tf.uniform_unit_scaling_initializer(), regularizer = l_regularizer)
    if use_weight_normalization: matrix = weight_normalization(matrix)
    matrix = tf.tile(tf.expand_dims(matrix, 0), [num_timesteps, 1, 1])

    res = tf.batch_matmul(input, matrix)

    if bias:
      bias_term = tf.get_variable(
          "Bias", [output_size],
          initializer=tf.constant_initializer(bias_start))
      res = res + bias_term

  if tranpose_input:
    res = tf.transpose(res, [1,0,2])

  return res