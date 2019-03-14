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
"""Tensorflow ray utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys
import tensorflow as tf


def triangulate(startpoints, endpoints, weights, name=None):
  """Triangulates 3d points by miminizing the sum of squared distances to rays.

  The rays are defined by their start points and endpoints. The rays should
  come from at least two views (cameras). The corresponding rays from each
  camera are stacked in dimension -2 of the 'startpoints' and 'endpoints'
  tensors. We choose the above metric over the standard reprojection-error
  metric, because this metric is can be optimized with a closed-form solve.

  Args:
    startpoints: N-D tensor of ray start points with shape [?, ..., V, 3], the
      number of views V should be greater or equal to 2, otherwise multiview
      triangulation is impossible.
    endpoints: N-D tensor of ray endpoints with shape [?, ..., V, 3], the number
      of views V should be greater or equal to 2, otherwise multiview
      triangulation is impossible. 'endpoints' tensor should have the same shape
      as 'startpoints' tensor.
    weights: (N-1)-D tensor of ray weights (certainties) with shape [?, ..., V].
      Weights should have all positive entries. Weight should have at least two
      non-zero entries for each point (at least two views should have
      certainties > 0).
    name: A name for this op. The default value of None means "ray_triangulate".

  Returns:
    points: (N-1)-D tensor points with shape [?, ..., 3].
  Raises:
    ValueError: If the shape of the arguments is not supported.
  """
  with tf.compat.v1.name_scope(name, "ray_triangulate",
                               [startpoints, endpoints, weights]):
    startpoints = tf.convert_to_tensor(value=startpoints)
    endpoints = tf.convert_to_tensor(value=endpoints)
    weights = tf.convert_to_tensor(value=weights)

    if startpoints.shape != endpoints.shape:
      raise ValueError("'startpoints' and 'endpoints' must have the same "
                       "shape.")
    if startpoints.shape[-1] != 3:
      raise ValueError("'startpoints' and 'endpoints' must have their last "
                       "dimension set to 3.")
    if len(startpoints.shape) < 2:
      raise ValueError("'startpoints' and 'endpoints' must have at least two "
                       "dimensions.")
    if startpoints.shape[-2] < 2:
      raise ValueError("'startpoints' and 'endpoints' must have their "
                       "penultimate dimension >= 2.")
    if weights.shape != startpoints.shape[:-1]:
      raise ValueError("'weights' should have the same shape as 'startpoints' "
                       "and 'endpoints', except that the last dimension should "
                       "not be present in 'weights'.")
    # TODO: verify that weights have all positive entries, verify
    # that weights have at least two non-zero entries for each point.

    startpoints = tf.debugging.check_numerics(
        startpoints, message="Inf or NaN detected in 'startpoints'")
    endpoints = tf.debugging.check_numerics(
        endpoints, message="Inf or NaN detected in 'startpoints'")
    weights = tf.debugging.check_numerics(
        weights, message="Inf or NaN detected in 'startpoints'")

    left_hand_side_list = []
    right_hand_side_list = []
    for camera_id in range(weights.shape[-1]):
      weights_singleview = weights[..., camera_id]
      startpoints_singleview = startpoints[..., camera_id, :]
      endpoints_singleview = endpoints[..., camera_id, :]

      ray = endpoints_singleview - startpoints_singleview
      ray = tf.nn.l2_normalize(ray, axis=-1)
      ray_x, ray_y, ray_z = tf.unstack(ray, axis=-1)

      zeros = tf.zeros_like(ray_x)
      cross_product_matrix = tf.stack(
          (zeros, -ray_z, ray_y, ray_z, zeros, -ray_x, -ray_y, ray_x, zeros),
          axis=-1)
      cross_product_matrix_shape = tf.concat(
          (tf.shape(input=cross_product_matrix)[:-1], (3, 3)), axis=-1)
      cross_product_matrix = tf.reshape(
          cross_product_matrix, shape=cross_product_matrix_shape)

      weights_singleview = tf.expand_dims(weights_singleview, axis=-1)
      weights_singleview = tf.expand_dims(weights_singleview, axis=-1)
      left_hand_side = weights_singleview * cross_product_matrix
      left_hand_side_list.append(left_hand_side)

      dot_product = tf.matmul(cross_product_matrix,
                              tf.expand_dims(startpoints_singleview, axis=-1))

      right_hand_side = weights_singleview * dot_product
      right_hand_side_list.append(right_hand_side)

    left_hand_side_multiview = tf.concat(left_hand_side_list, axis=-2)
    right_hand_side_multiview = tf.concat(right_hand_side_list, axis=-2)

    points = tf.linalg.lstsq(left_hand_side_multiview,
                             right_hand_side_multiview)
    points = tf.squeeze(points, axis=-1)

    return points


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
