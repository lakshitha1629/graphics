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
"""Tensorflow.graphics b-spline interpolation module.

  This module supports cardinal b-spline interpolation up to degree 4, with up
  to C3 smoothness. It has functions to calculate basis functions, control point
  weights, and the final interpolation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import enum
import numpy as np
import tensorflow as tf

from tensorflow_graphics.util import asserts


class Degree(enum.IntEnum):
  """Defines valid degrees for b-spline interpolation."""
  CONSTANT = 0
  LINEAR = 1
  QUADRATIC = 2
  CUBIC = 3
  QUARTIC = 4


def _constant(t):
  """B-Spline basis function of degree 0 for position t in range [0, 1]."""
  t = asserts.assert_all_in_range(t, 0.0, 1.0, open_bounds=False)

  # Piecewise constant - discontinuous
  return tf.expand_dims(tf.constant(1.0, shape=t.shape), axis=-1)


def _linear(t):
  """B-Spline basis functions of degree 1 for position t in range [0, 1]."""
  t = asserts.assert_all_in_range(t, 0.0, 1.0, open_bounds=False)

  # Piecewise linear - C0 smooth
  return tf.stack(((1.0 - t), t), axis=-1)


def _quadratic(t):
  """B-Spline basis functions of degree 2 for position t in range [0, 1]."""
  t = asserts.assert_all_in_range(t, 0.0, 1.0, open_bounds=False)

  # Piecewise quadratic - C1 smooth
  return tf.stack((tf.pow(
      (1.0 - t), 2.0) / 2.0, -tf.pow(t, 2.0) + t + 0.5, tf.pow(t, 2.0) / 2.0),
                  axis=-1)


def _cubic(t):
  """B-Spline basis functions of degree 3 for position t in range [0, 1]."""
  t = asserts.assert_all_in_range(t, 0.0, 1.0, open_bounds=False)

  # Piecewise cubic - C2 smooth
  return tf.stack(
      (tf.pow((1.0 - t), 3.0) / 6.0,
       (3.0 * tf.pow(t, 3.0) - 6.0 * tf.pow(t, 2.0) + 4.0) / 6.0,
       (-3.0 * tf.pow(t, 3.0) + 3.0 * tf.pow(t, 2.0) + 3.0 * t + 1.0) / 6.0,
       tf.pow(t, 3.0) / 6.0),
      axis=-1)


def _quartic(t):
  """B-Spline basis functions of degree 4 for position t in range [0, 1]."""
  t = asserts.assert_all_in_range(t, 0.0, 1.0, open_bounds=False)

  # Piecewise quartic - C3 smooth
  return tf.stack(
      (tf.pow((1.0 - t), 4.0) / 24.0,
       (-4.0 * tf.pow(1.0 - t, 4.0) + 4.0 * tf.pow(1.0 - t, 3.0) +
        6.0 * tf.pow(1.0 - t, 2.0) + 4.0 * (1.0 - t) + 1.0) / 24.0,
       (tf.pow(t, 4.0) - 2.0 * tf.pow(t, 3.0) - tf.pow(t, 2.0) + 2.0 * t) / 4.0
       + (11.0 / 24.0),
       (-4.0 * tf.pow(t, 4.0) + 4.0 * tf.pow(t, 3.0) + 6.0 * tf.pow(t, 2.0) +
        4.0 * t + 1.0) / 24.0, tf.pow(t, 4.0) / 24.0),
      axis=-1)


def knot_weights(positions,
                 num_knots,
                 degree,
                 cyclical,
                 sparse_mode=False,
                 name=None):
  """Function that converts cardinal b-spline positions to knot weights.

  Args:
    positions: Tensor with shape [batch_size, num_splines]. Positions must be
      between [0, num_knots-degree) for non-cyclical and [0, num_knots) for
      cyclical splines.
    num_knots: An integer, number of knots > 0.
    degree: integer degree of the spline, should be smaller than num_knots.
    cyclical: A boolean, whether the spline is cyclical.
    sparse_mode: A boolean, whether to return the nonzero weights only.
    name: A name for this op. Defaults to "bsplines_knot_weights".

  Returns:
    A tensor with dense weights for each control point, with the shape
    [batch_size, num_splines, num_knots] if sparse_mode is False.
    Otherwise returns a tensor of shape [batch_size, num_splines, degree + 1],
    and the corresponding shift amounts per sample, which is essentially
    tf.floor(positions). This can then be used to sparsely gather_nd from
    a spline.

  Raises:
    ValueError: If degree is larger than num_knots - 1.
    NotImplementedError: If degree > 4 or < 0
  """
  with tf.compat.v1.name_scope(name, 'bsplines_knot_weights', [positions]):
    positions = tf.convert_to_tensor(value=positions)
    if len(positions.get_shape().as_list()) != 2:
      raise ValueError(
          'Positions should have rank 2 with shape [batch_size, num_splines].')
    if degree > num_knots - 1:
      raise ValueError('Degree cannot be >= number of knots.')
    if degree > 4 or degree < 0:
      raise NotImplementedError('Degree should be between 0 and 4.')

    all_basis_functions = {
        # Maps valid degrees to functions.
        Degree.CONSTANT: _constant,
        Degree.LINEAR: _linear,
        Degree.QUADRATIC: _quadratic,
        Degree.CUBIC: _cubic,
        Degree.QUARTIC: _quartic
    }
    basis_functions = all_basis_functions[degree]

    if not cyclical and num_knots - degree == 1:
      # Already dense and in order, no need to shift.
      return basis_functions(positions)

    # Calculates the degree + 1 sparse knot weights using decimal parts.
    shift = tf.floor(positions)
    sparse_weights = basis_functions(positions - shift)
    shift = tf.cast(shift, tf.int32)

    if sparse_mode:
      # Returns just the weights and the shift amounts, so that tf.gather_nd on
      # the knots can be used to sparsely activate knots.
      return sparse_weights, shift

    # Scatters the sparse weights into a dense matrix by taking cyclical splines
    # into account. A spline is cyclical if positions > num_knots - degree
    num_samples, num_splines = positions.get_shape().as_list()
    shape = tf.constant([num_samples, num_splines, num_knots])

    # Fixed indexing with meshgrid for the first 2 dimensions of scatter_nd.
    x, y, z = np.meshgrid(
        np.arange(num_samples, dtype=np.int32),
        np.arange(num_splines, dtype=np.int32),
        np.arange(degree + 1, dtype=np.int32),
        indexing='ij')
    col_x, col_y, col_z = [tf.constant(np.reshape(d, (-1,))) for d in [x, y, z]]
    shifts_col = tf.reshape(
        tf.tile(tf.expand_dims(shift, -1), (1, 1, degree + 1)), (-1,))
    shifted_col = col_z + shifts_col
    if cyclical:
      shifted_col = tf.mod(shifted_col, num_knots)
    out = tf.stack((col_x, col_y, shifted_col), axis=-1)
    indices = tf.reshape(out, shape=(num_samples, num_splines, degree + 1, 3))
    return tf.scatter_nd(indices, sparse_weights, shape)


def interpolate_with_weights(knots, weights, name=None):
  """Interpolates knots using knot weights.

  Args:
    knots: N-D tensor with rank >=1 and shape [dim_1, ..., dim_j, num_knots]
    weights: Tensor with shape [batch_size, num_splines, num_knots] containing
      dense weights for knots.
    name: A name for this op. Defaults to "bsplines_interpolate_with_weights".

  Returns:
    N-D tensor with shape [batch_size, dim_1, ..., dim_j], which is the result
      of spline interpolation.

  Raises:
    ValueError: If last dimensions of knots and weights are not equal.
  """
  with tf.compat.v1.name_scope(name, 'bsplines_interpolate_with_weights',
                               [knots, weights]):
    knots = tf.convert_to_tensor(value=knots)
    weights = tf.convert_to_tensor(value=weights)
    if knots.get_shape().as_list()[-1] != weights.get_shape().as_list()[-1]:
      raise ValueError('Last dimensions should match.')
  return tf.tensordot(weights, knots, (-1, -1))


def interpolate(knots, positions, degree, cyclical, name=None):
  """Applies b-spline interpolation to input control points (knots).

  Args:
    knots: N-D tensor with rank >=1 and shape [dim_1, ..., dim_j, num_knots]
    positions: Tensor with shape [batch_size, num_splines]. Positions must be
      between [0, num_knots-degree) for non-cyclical and [0, num_knots) for
      cyclical splines.
    degree: An integer between 0 and 4, or an enumerated constant from the
      Degree class, which is the degree of the splines.
    cyclical: A boolean, whether the splines are cyclical.
    name: A name for this op. Defaults to "bspline_interpolate".

  Returns:
    N-D tensor with shape [batch_size, dim_1, ..., dim_j], which is the result
      of spline interpolation.
  """
  with tf.compat.v1.name_scope(name, 'bspline_interpolate', [knots, positions]):
    knots = tf.convert_to_tensor(value=knots)
    positions = tf.convert_to_tensor(value=positions)
    num_knots = knots.get_shape().as_list()[-1]
    weights = knot_weights(positions, num_knots, degree, cyclical, False, name)
    return interpolate_with_weights(knots, weights)


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) or
    inspect.isclass(obj) and not obj_name.startswith('_')
]
