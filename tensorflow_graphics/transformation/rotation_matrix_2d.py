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
"""This module implements TensorFlow 2d rotation matrix utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import tensorflow as tf

from tensorflow_graphics.transformation import rotation_matrix_common


def from_euler(angle, name=None):
  """Convert an angle to a 2d rotation matrix.

  Args:
    angle: N-D tensor of shape `[?, ..., ?, 1]`.
    name: A name for this op. Defaults to "rotation_matrix_2d_from_angle".

  Returns:
    (N+1)-D tensor of shape `[?, ..., ?, 2, 2]`.

  Raises:
    ValueError: if the shape of `angle` is not supported.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_2d_from_angle", [angle]):
    angle = tf.convert_to_tensor(value=angle)
    shape = angle.shape.as_list()
    if shape[-1] != 1:
      raise ValueError("'angle' must have 1 dimension.")

    cos_angle = tf.cos(angle)
    sin_angle = tf.sin(angle)
    # pyformat: disable
    matrix = tf.stack((cos_angle, -sin_angle,
                       sin_angle, cos_angle),
                      axis=-1)
    # pyformat: enable
    output_shape = tf.concat((tf.shape(input=angle)[:-1], (2, 2)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


def from_euler_with_small_angles_approximation(angles, name=None):
  """Under the small angle assumption, convert an angle to a 2d rotation matrix.

  Args:
    angles: N-D tensor of shape `[?, ..., ?, 1]` containing angles in radians.
    name: A name for this op. Defaults to
      "rotation_matrix_2d_from_euler_with_small_angles_approximation".

  Returns:
    (N+1)-D tensor of shape `[?, ..., ?, 2, 2]` containing the resulting
    rotation matrix.

  Raises:
    ValueError: if the shape of `angle` is not supported.
  """
  with tf.compat.v1.name_scope(
      name, "rotation_matrix_2d_from_euler_with_small_angles_approximation",
      [angles]):
    angles = tf.convert_to_tensor(value=angles)
    shape = angles.shape.as_list()
    if shape[-1] != 1:
      raise ValueError("'angles' must have 1 dimension.")

    cos_angle = 1.0 - 0.5 * angles * angles
    sin_angle = angles
    # pyformat: disable
    matrix = tf.stack((cos_angle, -sin_angle,
                       sin_angle, cos_angle),
                      axis=-1)
    # pyformat: enable
    output_shape = tf.concat((tf.shape(input=angles)[:-1], (2, 2)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


def inverse(matrix, name=None):
  """Computes the inverse of a 2D rotation matrix.

  Args:
    matrix: N-D tensor of shape `[?, ..., ?, 2, 2]`.
    name: A name for this op. Defaults to "rotation_matrix_2d_inverse".

  Returns:
    A N-D tensor of shape `[?, ..., ?, 2, 2]`.

  Raises:
    ValueError: if the shape of `matrix` is not supported.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_2d_inverse", [matrix]):
    matrix = tf.convert_to_tensor(value=matrix)
    shape_matrix = matrix.shape.as_list()
    if shape_matrix[-2:] != [2, 2]:
      raise ValueError("'matrix' must have 2x2 dimensions.")
    input_len = len(shape_matrix)
    perm = list(range(input_len - 2)) + [input_len - 1, input_len - 2]
    return tf.transpose(a=matrix, perm=perm)


def is_normalized(rotation_matrix, atol=1e-3, name=None):
  """Determines if rotation_matrix is a valid rotation matrix.

  Args:
    rotation_matrix: N-D tensor of shape `[?, ..., ?, 2, 2]`.
    atol: Absolute tolerance parameter.
    name: A name for this op. Defaults to "rotation_matrix_2d_is_normalized".

  Returns:
    A N-D tensor of shape `[?, ..., ?, 1]` where False indicates that the
    input is not a valid rotation matrix.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_2d_is_normalized",
                               [rotation_matrix]):
    rotation_matrix = tf.convert_to_tensor(value=rotation_matrix)
    shape = rotation_matrix.shape.as_list()
    shape_length = len(shape)
    if shape_length < 2 or shape[-1] != 2 or shape[-2] != 2:
      raise ValueError("'rotation_matrix_2d' must have 2x2 dimensions.")
    return rotation_matrix_common.is_normalized(rotation_matrix, atol)


def rotate(point, matrix, name=None):
  """Rotate a point using a rotation matrix 2d.

  Args:
    point: N-D tensor of shape `[?, ..., ?, 2]`.
    matrix: N-D tensor of shape `[?, ..., ?, 2,2]`.
    name: A name for this op. Defaults to "rotation_matrix_2d_rotate".

  Returns:
    A N-D tensor of shape `[?, ..., ?, 2]`.

  Raises:
    ValueError: if the shape of `point` or `matrix` is not supported.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_2d_rotate",
                               [point, matrix]):
    point = tf.convert_to_tensor(value=point)
    matrix = tf.convert_to_tensor(value=matrix)
    shape_point = point.shape.as_list()
    shape_matrix = matrix.shape.as_list()
    if shape_point[-1] != 2:
      raise ValueError("'point' must have 2 dimensions.")
    if shape_matrix[-2:] != [2, 2]:
      raise ValueError("'matrix' must have 2x2 dimensions.")
    point = tf.convert_to_tensor(value=point)
    matrix = tf.convert_to_tensor(value=matrix)
    point = tf.expand_dims(point, axis=-1)
    rotated_point = tf.matmul(matrix, point)
    return tf.squeeze(rotated_point, axis=-1)


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
