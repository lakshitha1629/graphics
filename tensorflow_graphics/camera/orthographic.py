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
"""This module implements orthographic camera utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import tensorflow as tf


def project(point_3d, name=None):
  """Projects a 3d point on the 2d camera plane.

  Args:
    point_3d: The 3d point to project. N-D tensor of shape `[?, ..., ?, 3]`.
    name: A name for this op. Defaults to "orthographic_project".

  Returns:
    The projected 2d point. N-D tensor of shape `[?, ..., ?, 2]`.

  Raises:
    ValueError: if the shape of `point_3d` is not supported.
  """
  with tf.compat.v1.name_scope(name, "orthographic_project", [point_3d]):
    point_3d = tf.convert_to_tensor(value=point_3d)
    shape_point_3d = point_3d.shape.as_list()
    if shape_point_3d[-1] != 3:
      raise ValueError("'point_3d' must have 3 dimensions.")
    return point_3d[..., :2]


def ray(point_2d, name=None):
  """Computes the ray for a 2d point.

  Args:
    point_2d: The 2d point. N-D tensor of shape `[?, ..., ?, 2]`.
    name: A name for this op. Defaults to "orthographic_ray".

  Returns:
    The 3d ray. N-D tensor of shape `[?, ..., ?, 3]`.

  Raises:
    ValueError: if the shape of `point_2d` is not supported.
  """
  with tf.compat.v1.name_scope(name, "orthographic_ray", [point_2d]):
    point_2d = tf.convert_to_tensor(value=point_2d)
    if point_2d.get_shape()[-1] != 2:
      raise ValueError("'point_2d' must have 2 dimensions.")
    zeros = tf.zeros_like(point_2d)
    ones = tf.ones_like(point_2d[..., :1])
    zeros = tf.zeros_like(point_2d)
    # The multiplication of point_2d by zeros is necessary for getting
    # gradients.
    return tf.concat((point_2d * zeros, ones), axis=-1)


def unproject(point_2d, depth, name=None):
  """Unprojects a 2d point in 3d.

  Args:
    point_2d: The 2d point to unproject. N-D tensor of shape `[?, ..., ?, 2]`.
    depth: The depth for this 2d point. N-D tensor of shape `[?, ..., ?, 1]`.
    name: A name for this op. Defaults to "orthographic_unproject".

  Returns:
    The unprojected 3d point. N-D tensor of shape `[?, ..., ?, 3]`.

  Raises:
    ValueError: if the shape of `point_2d`, `depth` is not supported.
  """
  with tf.compat.v1.name_scope(name, "orthographic_unproject",
                               [point_2d, depth]):
    point_2d = tf.convert_to_tensor(value=point_2d)
    depth = tf.convert_to_tensor(value=depth)
    if point_2d.get_shape()[-1] != 2:
      raise ValueError("'point_2d' must have 2 dimensions.")
    if depth.get_shape()[-1] != 1:
      raise ValueError("'depth' must have 1 dimension.")
    return tf.concat((point_2d, depth), axis=-1)


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
