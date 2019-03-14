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
"""Tensorflow point utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import tensorflow as tf

from tensorflow_graphics.util import asserts


def _check_shapes(point, origin, direction):
  """Checks if input shapes are as expected."""
  shape_point = point.shape.as_list()
  shape_origin = origin.shape.as_list()
  shape_direction = direction.shape.as_list()
  if shape_point[-1] != shape_origin[-1] or shape_point[-1] != shape_direction[
      -1]:
    raise ValueError(
        "'point', 'origin', and 'direction' must have the same dimensions.")


def distance_to_ray(point, origin, direction, name=None):
  """Computes distance from a point to a ray.

  Args:
    point: N-D tensor of shape `[?, ..., ?, M]`.
    origin: N-D tensor of shape `[?, ..., ?, M]`.
    direction: N-D tensor of shape `[?, ..., ?, M]`. The direction vector needs
      to be normalized.
    name: A name for this op. Defaults to "point_distance_to_ray".

  Returns:
    N-D tensor of shape `[?, ..., ?, 1]` containing the distance from the point
    to the corresponding ray.

  Raises:
    ValueError: If the shape of `point`, `origin`, or 'direction' is not
    supported.
  """
  with tf.compat.v1.name_scope(name, "point_distance_to_ray",
                               [point, origin, direction]):
    point = tf.convert_to_tensor(value=point)
    origin = tf.convert_to_tensor(value=origin)
    direction = tf.convert_to_tensor(value=direction)
    _check_shapes(point, origin, direction)

    direction = asserts.assert_normalized(direction)
    vec = point - origin
    dot = tf.reduce_sum(input_tensor=vec * direction, axis=-1, keepdims=True)
    vec -= dot * direction
    return tf.norm(tensor=vec, axis=-1, keepdims=True)


def project_to_ray(point, origin, direction, name=None):
  """Computes the projection of a point on a ray.

  Args:
    point: N-D tensor of shape `[?, ..., ?, M]`.
    origin: N-D tensor of shape `[?, ..., ?, M]`.
    direction: N-D tensor of shape `[?, ..., ?, M]`. The direction vector needs
      to be normalized.
    name: A name for this op. Defaults to "point_project_to_ray".

  Returns:
    N-D tensor of shape `[?, ..., ?, M]` containing the projected point.

  Raises:
    ValueError: If the shape of `point`, `origin`, or 'direction' is not
    supported.
  """
  with tf.compat.v1.name_scope(name, "point_project_to_ray",
                               [point, origin, direction]):
    point = tf.convert_to_tensor(value=point)
    origin = tf.convert_to_tensor(value=origin)
    direction = tf.convert_to_tensor(value=direction)
    _check_shapes(point, origin, direction)

    direction = asserts.assert_normalized(direction)
    vec = point - origin
    dot = tf.reduce_sum(input_tensor=vec * direction, axis=-1, keepdims=True)
    return origin + dot * direction


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
