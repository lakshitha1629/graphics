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
"""This module implements weighted interpolation for point sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import tensorflow as tf

from tensorflow_graphics.geometry import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import safe_ops


def interpolate(points,
                weights,
                indices,
                normalize=True,
                allow_negative_weights=False,
                name=None):
  """Weighted interpolation for N-D point sets.

  Given an N-D point set, this function can be used to generate a new point set
  that is formed by interpolating a subset of points in the set.

  Args:
    points: tf.Tensor with shape [?, ..., ?, N] and rank R > 1, where N is the
      dimensionality of the points.
    weights: tf.Tensor with shape [dim_1, ..., dim_k, M], where M is the number
      of points to interpolate for each output point. k can be zero.
    indices: tf.Tensor of dtype tf.int32 and shape [dim_1, ..., dim_k, M, R-1],
      which contains the point indices to be used for each output point. The R-1
      dimensional axis gives the slice index of a single point in 'points'. The
      first k+1 dimensions of weights and indices must match, or be broadcast
      compatible.
    normalize: Boolean, whether to normalize the weights on the last axis.
    allow_negative_weights: Boolean, whether negative weights are allowed.
    name: A name for this op. Defaults to "weighted_interpolate".

  Returns:
    tf.Tensor with shape [dim_1, ..., dim_k, N], which are the
    interpolated N-D points. The first k dimensions will be the same as weights
    and indices.
  """
  with tf.compat.v1.name_scope(name, "weighted_interpolate",
                               [points, weights, indices]):
    points = tf.convert_to_tensor(value=points)
    weights = tf.convert_to_tensor(value=weights)
    indices = tf.convert_to_tensor(value=indices)

    if not allow_negative_weights:
      weights = asserts.assert_all_above(weights, 0.0, open_bound=False)

    if normalize:
      sums = tf.reduce_sum(input_tensor=weights, axis=-1, keepdims=True)
      sums = asserts.assert_nonzero_norm(sums)
      weights = safe_ops.safe_signed_div(weights, sums)

    point_lists = tf.gather_nd(points, indices)
    return vector.dot(
        point_lists, tf.expand_dims(weights, axis=-1), axis=-2, keepdims=False)


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
