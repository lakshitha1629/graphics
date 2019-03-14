#Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Assert functions for tf.graphics modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import numpy as np
import tensorflow as tf

from tensorflow_graphics.util import tfg_flags

FLAGS = flags.FLAGS


def assert_all_above(vector, minval, open_bound=False, name=None):
  """Checks whether all values of vector are above minval.

  Args:
    vector: N-D tensor with any shape.
    minval: A scalar or an N-D tensor representing the lower bound.
    open_bound: A boolean indicating whether the range is open or closed.
    name: A name for this op. Defaults to 'assert_all_above'.

  Raises:
    tf.errors.InvalidArgumentError: if any entry of the input is below `minval`.

  Returns:
    The input vector, with dependence on the assertion operator in the graph.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return vector

  with tf.compat.v1.name_scope(name, 'assert_all_above', [vector, minval]):
    vector = tf.convert_to_tensor(value=vector)
    minval = tf.convert_to_tensor(value=minval, dtype=vector.dtype)
    if open_bound:
      assert_ops = (tf.compat.v1.assert_greater(vector, minval),)
    else:
      assert_ops = (tf.compat.v1.assert_greater_equal(vector, minval),)
    with tf.control_dependencies(assert_ops):
      return tf.identity(vector)


def assert_all_below(vector, maxval, open_bound=False, name=None):
  """Checks whether all values of vector are below maxval.

  Args:
    vector: N-D tensor with any shape.
    maxval: A scalar or an N-D tensor representing the upper bound.
    open_bound: A boolean indicating whether the range is open or closed.
    name: A name for this op. Defaults to 'assert_all_below'.

  Raises:
    tf.errors.InvalidArgumentError: if any entry of the input exceeds `maxval`.

  Returns:
    The input vector, with dependence on the assertion operator in the graph.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return vector

  with tf.compat.v1.name_scope(name, 'assert_all_below', [vector, maxval]):
    vector = tf.convert_to_tensor(value=vector)
    maxval = tf.convert_to_tensor(value=maxval, dtype=vector.dtype)
    if open_bound:
      assert_ops = (tf.compat.v1.assert_less(vector, maxval),)
    else:
      assert_ops = (tf.compat.v1.assert_less_equal(vector, maxval),)
    with tf.control_dependencies(assert_ops):
      return tf.identity(vector)


def assert_all_in_range(vector, minval, maxval, open_bounds=False, name=None):
  """Checks whether all values of vector are between minval and maxval.

  This function checks if all the values in the given vector are in an interval
  [minval, maxval] if open_bounds is False, or in ]minval, maxval[ if it is set
  to True.

  Args:
    vector: N-D tensor with any shape.
    minval: A float or an N-D tensor. Lower bound as a float or as a tensor with
      the same shape as vector.
    maxval: A float or an N-D tensor. Upper bound as a float or as a tensor with
      the same shape as vector.
    open_bounds: A boolean indicating whether the range is open or closed.
    name: A name for this op. Defaults to 'assert_all_in_range'.

  Raises:
    tf.errors.InvalidArgumentError: If vector is not in the expected range.

  Returns:
    The input vector, with dependence on the assertion operator in the graph.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return vector

  with tf.compat.v1.name_scope(name, 'assert_all_in_range',
                               [vector, minval, maxval]):
    vector = tf.convert_to_tensor(value=vector)
    minval = tf.convert_to_tensor(value=minval, dtype=vector.dtype)
    maxval = tf.convert_to_tensor(value=maxval, dtype=vector.dtype)
    if open_bounds:
      assert_ops = (tf.compat.v1.assert_less(vector, maxval),
                    tf.compat.v1.assert_greater(vector, minval))
    else:
      assert_ops = (tf.compat.v1.assert_less_equal(vector, maxval),
                    tf.compat.v1.assert_greater_equal(vector, minval))
    with tf.control_dependencies(assert_ops):
      return tf.identity(vector)


def assert_nonzero_norm(vector, eps=None, name=None):
  """Checks whether vector/quaternion has non-zero norm in its last dimension.

  This function checks whether all the norms of the vectors are greater than
  eps, such that normalizing them will not generate NaN values. Normalization is
  assumed to be done in the last dimension of vector. If eps is left as None,
  the function will determine the most suitable value depending on the dtype of
  the vector.

  Args:
    vector: N-D tensor with any shape.
    eps: A float. Tolerance to determine if the norm is equal to zero.
    name: A name for this op. Defaults to 'assert_nonzero_norm'.

  Raises:
    tf.errors.InvalidArgumentError: If vector is not safe to normalize.

  Returns:
    The input vector, with dependence on the assertion operator in the graph.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return vector

  with tf.compat.v1.name_scope(name, 'assert_nonzero_norm', [vector, eps]):
    vector = tf.convert_to_tensor(value=vector)
    norm = tf.norm(tensor=vector, axis=-1)
    if eps is None:
      # Select eps for division, since the vector will be divided by the norm.
      eps = select_eps_for_division(vector.dtype)
    eps = tf.convert_to_tensor(value=eps, dtype=vector.dtype)
    with tf.control_dependencies([tf.compat.v1.assert_greater(norm, eps)]):
      return tf.identity(vector)


def assert_normalized(vector, eps=None, name=None):
  """Checks whether vector/quaternion is normalized in its last dimension.

  Args:
    vector: N-D tensor with shape [?, ..., ?, M].
    eps: A float. Tolerance to determine if the norm is equal to 1.0.
    name: A name for this op. Defaults to 'assert_normalized'.

  Raises:
    tf.errors.InvalidArgumentError: If vector is not normalized.

  Returns:
    The input vector, with dependence on the assertion operator in the graph.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return vector

  with tf.compat.v1.name_scope(name, 'assert_normalized', [vector, eps]):
    vector = tf.convert_to_tensor(value=vector)
    norm = tf.norm(tensor=vector, axis=-1)
    if eps is None:
      eps = select_eps_for_addition(norm.dtype)
    eps = tf.convert_to_tensor(value=eps, dtype=norm.dtype)
    one = tf.constant(1.0, dtype=norm.dtype)
    with tf.control_dependencies(
        [tf.compat.v1.assert_near(norm, one, atol=eps)]):
      return tf.identity(vector)


def select_eps_for_addition(dtype):
  """Returns 2 * machine epsilon based on dtype.

  This function picks an epsilon slightly greater than the machine epsilon,
  which is the upper bound on relative error. This value ensures that
  1.0 + eps != 1.0.

  Args:
    dtype: tf.DType. DType of the tensor to which eps will be added.

  Raises:
    ValueError: If dtype is not a floating type.

  Returns:
    A float to be used to make operations safe, i.e. to prevent NaN or Inf.
  """
  return 2.0 * np.finfo(dtype.as_numpy_dtype()).eps


def select_eps_for_division(dtype):
  """Selects default values for epsilon to make divisions safe based on dtype.

  This function returns an epsilon slightly greater than the smallest positive
  floating number that is representable for the given dtype. This is mainly used
  to prevent division by zero, which produces Inf values. However, if the
  nominator is orders of magnitude greater than 1.0, eps should also be
  increased accordingly. Only floating types are supported.

  Args:
    dtype: tf.DType. DType of the tensor to which eps will be added.

  Raises:
    ValueError: If dtype is not a floating type.

  Returns:
    A float to be used to make operations safe, i.e. to prevent NaN or Inf.
  """
  return 4.0 * np.finfo(dtype.as_numpy_dtype()).tiny
