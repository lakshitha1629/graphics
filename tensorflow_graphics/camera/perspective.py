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
r"""This module implements TensorFlow perspective camera utility functions.

The perspective camera is defined using a focal length $$(f_x, f_y)$$ and
a principal point $$(c_x, c_y)$$. It can be written as a calibration matrix

$$
\begin{bmatrix}
f_x & 0 & c_x \\
0  & f_y & c_y \\
0  & 0  & 1 \\
\end{bmatrix}
$$

The focal length and the principal point are tensors of shape `[?, ..., ?, 2]`.
Note that the current implementation does not take into account distortion or
skew parameters.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import tensorflow as tf


def intrinsics_from_matrix(matrix, name=None):
  """Extracts intrinsic parameters from a calibration matrix.

  Args:
    matrix: The calibration matrix. N-D tensor of shape `[?, ..., ?, 3, 3]`.
    name: A name for this op. Defaults to "perspective_intrinsics_from_matrix".

  Returns:
    The focal and principal point. N-D tensors of shape `[?, ..., ?, 2]`.

  Raises:
    ValueError: if the shape of `matrix` is not supported.
  """
  with tf.compat.v1.name_scope(name, "perspective_intrinsics_from_matrix",
                               [matrix]):
    matrix = tf.convert_to_tensor(value=matrix)
    shape_matrix = matrix.shape.as_list()
    if shape_matrix[-2:] != [3, 3]:
      raise ValueError("'matrix' must have 3x3 dimensions.")

    fx = matrix[..., 0, 0]
    fy = matrix[..., 1, 1]
    cx = matrix[..., 0, 2]
    cy = matrix[..., 1, 2]
    focal = tf.stack((fx, fy), axis=-1)
    principal_point = tf.stack((cx, cy), axis=-1)
  return focal, principal_point


def matrix_from_intrinsics(focal, principal_point, name=None):
  """Builds calibration matrix from intrinsic parameters.

  Args:
    focal: The focal length. N-D tensor of shape `[?, ..., ?, 2]`.
    principal_point: The principal point. N-D tensor of shape `[?, ..., ?, 2]`.
    name: A name for this op. Defaults to "perspective_matrix_from_intrinsics".

  Returns:
    The calibration matrix. N-D tensor of shape `[?, ..., ?, 3, 3]`.

  Raises:
    ValueError: if the shape of `focal`, or `principal_point` is not
    supported.
  """
  with tf.compat.v1.name_scope(name, "perspective_matrix_from_intrinsics",
                               [focal, principal_point]):
    focal = tf.convert_to_tensor(value=focal)
    principal_point = tf.convert_to_tensor(value=principal_point)
    shape_focal = focal.shape.as_list()
    shape_principal_point = principal_point.shape.as_list()
    if shape_focal[-1] != 2:
      raise ValueError("'focal' must have 2 dimensions.")
    if shape_principal_point[-1] != 2:
      raise ValueError("'principal_point' must have 2 dimensions.")

    fx, fy = tf.unstack(focal, axis=-1)
    cx, cy = tf.unstack(principal_point, axis=-1)
    zero = tf.zeros_like(fx)
    one = tf.ones_like(fx)
    # pyformat: disable
    matrix = tf.stack((fx, zero, cx,
                       zero, fy, cy,
                       zero, zero, one),
                      axis=-1)
    # pyformat: enable
    matrix_shape = tf.shape(input=matrix)
    output_shape = tf.concat((matrix_shape[:-1], (3, 3)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


def project(point_3d, focal, principal_point, name=None):
  """Projects a 3d point on the 2d camera plane.

  Args:
    point_3d: The 3d point to project. N-D tensor of shape `[?, ..., ?, 3]`.
    focal: The focal length. N-D tensor of shape `[?, ..., ?, 2]`.
    principal_point: The principal point. N-D tensor of shape `[?, ..., ?, 2]`.
    name: A name for this op. Defaults to "perspective_project".

  Returns:
    The projected 2d point. N-D tensor of shape `[?, ..., ?, 2]`.

  Raises:
    ValueError: if the shape of `point_3d`, `focal`, or `principal_point` is not
    supported.
  """
  with tf.compat.v1.name_scope(name, "perspective_project",
                               [point_3d, focal, principal_point]):
    point_3d = tf.convert_to_tensor(value=point_3d)
    focal = tf.convert_to_tensor(value=focal)
    principal_point = tf.convert_to_tensor(value=principal_point)
    shape_point_3d = point_3d.shape.as_list()
    shape_focal = focal.shape.as_list()
    shape_principal_point = principal_point.shape.as_list()
    if shape_point_3d[-1] != 3:
      raise ValueError("'point_3d' must have 3 dimensions.")
    if shape_focal[-1] != 2:
      raise ValueError("'focal' must have 2 dimensions.")
    if shape_principal_point[-1] != 2:
      raise ValueError("'principal_point' must have 2 dimensions.")

    point_2d, depth = tf.split(point_3d, (2, 1), axis=-1)
    eps = sys.float_info.epsilon
    depth = tf.sign(depth) * tf.maximum(tf.abs(depth), eps)
    point_2d *= focal / depth
    point_2d += principal_point
  return point_2d


def ray(point_2d, focal, principal_point, name=None):
  """Computes the ray for a 2d point (the z compoment of the ray is 1).

  Args:
    point_2d: The 2d point. N-D tensor of shape `[?, ..., ?, 2]`.
    focal: The focal length. N-D tensor of shape `[?, ..., ?, 2]`.
    principal_point: The principal point. N-D tensor of shape `[?, ..., ?, 2]`.
    name: A name for this op. Defaults to "perspective_ray".

  Returns:
    The 3d ray. N-D tensor of shape `[?, ..., ?, 3]`.

  Raises:
    ValueError: if the shape of `point_2d`, `focal`, or `principal_point` is not
    supported.
  """
  with tf.compat.v1.name_scope(name, "perspective_ray",
                               [point_2d, focal, principal_point]):
    point_2d = tf.convert_to_tensor(value=point_2d)
    focal = tf.convert_to_tensor(value=focal)
    principal_point = tf.convert_to_tensor(value=principal_point)
    shape_point_2d = point_2d.shape.as_list()
    shape_focal = focal.shape.as_list()
    shape_principal_point = principal_point.shape.as_list()
    if shape_point_2d[-1] != 2:
      raise ValueError("'point_2d' must have 2 dimensions.")
    if shape_focal[-1] != 2:
      raise ValueError("'focal' must have 2 dimensions.")
    if shape_principal_point[-1] != 2:
      raise ValueError("'principal_point' must have 2 dimensions.")

    point_2d -= principal_point
    eps = sys.float_info.epsilon
    focal = tf.sign(focal) * tf.maximum(tf.abs(focal), eps)
    point_2d /= focal
    padding = [[0, 0] for _ in shape_point_2d]
    padding[-1][-1] = 1
    return tf.pad(
        tensor=point_2d, paddings=padding, mode="CONSTANT", constant_values=1.0)


def unproject(point_2d, depth, focal, principal_point, name=None):
  """Unprojects a 2d point in 3d.

  Args:
    point_2d: The 2d point to unproject. N-D tensor of shape `[?, ..., ?, 2]`.
    depth: The depth for this 2d point. N-D tensor of shape `[?, ..., ?, 1]`.
    focal: The focal length. N-D tensor of shape `[?, ..., ?, 2]`.
    principal_point: The principal point. N-D tensor of shape `[?, ..., ?, 2]`.
    name: A name for this op. Defaults to "perspective_unproject".

  Returns:
    The unprojected 3d point. N-D tensor of shape `[?, ..., ?, 3]`.

  Raises:
    ValueError: if the shape of `point_2d`, `depth`, `focal`, or
    `principal_point` is not supported.
  """
  with tf.compat.v1.name_scope(name, "perspective_unproject",
                               [point_2d, depth, focal, principal_point]):
    point_2d = tf.convert_to_tensor(value=point_2d)
    depth = tf.convert_to_tensor(value=depth)
    focal = tf.convert_to_tensor(value=focal)
    principal_point = tf.convert_to_tensor(value=principal_point)
    shape_point_2d = point_2d.shape.as_list()
    shape_depth = depth.shape.as_list()
    shape_focal = focal.shape.as_list()
    shape_principal_point = principal_point.shape.as_list()
    if shape_point_2d[-1] != 2:
      raise ValueError("'point_2d' must have 2 dimensions.")
    if shape_depth[-1] != 1:
      raise ValueError("'depth' must have 1 dimension.")
    if shape_focal[-1] != 2:
      raise ValueError("'focal' must have 2 dimensions.")
    if shape_principal_point[-1] != 2:
      raise ValueError("'principal_point' must have 2 dimensions.")

    point_2d -= principal_point
    eps = sys.float_info.epsilon
    focal = tf.sign(focal) * tf.maximum(tf.abs(focal), eps)
    point_2d *= depth / focal
    return tf.concat((point_2d, depth), axis=-1)


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
