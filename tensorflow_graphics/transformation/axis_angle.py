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
"""This module implements TensorFlow axis-angle utility functions.

The axis-angle representation is defined using a three dimensional axis
$$[x, y, z]^T$$ and an angle leading to tensors `[?, ..., ?, 3]` and
`[?, ..., ?, 1]`. Note that some so the functions expect normalized axis as
inputs where $$x^2 + y^2 + z^2 = 1$$.

More details about the axis-angle formalism can be found on [this page.]
(../../../../complementary_docs/transformation/axis_angle.md)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import tensorflow as tf

from tensorflow_graphics.transformation import quaternion as quaternion_lib
from tensorflow_graphics.transformation import rotation_matrix_3d
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import safe_ops


def from_euler(angles, name=None):
  """Converts Euler angles to an axis angle representation.

  The conversion is performed by first converting to a quaternion
  representation, and then by converting the quaternion to an axis-angle.

  Args:
    angles: N-D tensor of shape `[?, ..., ?, 3]`.
    name: A name for this op. Defaults to "axis_angle_from_euler".

  Returns:
    Tuple of two N-D tensors of shape `[?, ..., ?, 3]` and `[?, ..., ?, 1]`. The
    first tensor represent the axis, and the second one the angle. The resulting
    axis-angle is normalized.
  """
  with tf.compat.v1.name_scope(name, "axis_angle_from_euler", [angles]):
    quaternion = quaternion_lib.from_euler(angles)
    return from_quaternion(quaternion)


def from_euler_with_small_angles_approximation(angles, name=None):
  """Converts small Euler angles to an axis angle representation.

  The conversion is performed by first converting to a quaternion
  representation, and then by converting the quaternion to an axis-angle.

  Args:
    angles: N-D tensor of shape `[?, ..., ?, 3]`.
    name: A name for this op. Defaults to
      "axis_angle_from_euler_with_small_angles_approximation".

  Returns:
    Tuple of two N-D tensors of shape `[?, ..., ?, 3]` and `[?, ..., ?, 1]`. The
    first tensor represent the axis, and the second one the angle. The resulting
    axis-angle is normalized.
  """
  with tf.compat.v1.name_scope(
      name, "axis_angle_from_euler_with_small_angles_approximation", [angles]):
    quaternion = quaternion_lib.from_euler_with_small_angles_approximation(
        angles)
    # We need to normalize the quaternion due to the small angle approximation.
    quaternion = tf.nn.l2_normalize(quaternion, axis=-1)
    return from_quaternion(quaternion)


def from_quaternion(quaternion, name=None):
  """Convert a quaternion to an axis angle representation.

  Args:
    quaternion: N-D tensor of shape `[?, ..., ?, 4]`, a tensor of normalized
      quaternions.
    name: A name for this op. Defaults to "axis_angle_from_quaternion".

  Returns:
    Tuple of two N-D tensors of shape `[?, ..., ?, 3]` and `[?, ..., ?, 1]`.
    The first tensor represent the axis, and the second one the angle. The
    resulting axis-angle is normalized.

  Raises:
    ValueError: if the shape of `axis` or `angles` is not supported.
  """
  with tf.compat.v1.name_scope(name, "axis_angle_from_quaternion",
                               [quaternion]):
    quaternion = tf.convert_to_tensor(value=quaternion)
    shape = quaternion.shape.as_list()
    if shape[-1] != 4:
      raise ValueError("'quaternion' must have 4 dimensions.")

    quaternion = asserts.assert_normalized(quaternion)
    xyz, w = tf.split(quaternion, (3, 1), axis=-1)
    norm = tf.norm(tensor=xyz, axis=-1, keepdims=True)
    angle = 2.0 * tf.atan2(
        norm,
        tf.abs(w) + asserts.select_eps_for_addition(quaternion.dtype))
    axis_general_case = safe_ops.safe_unsigned_div(
        safe_ops.nonzero_sign(w) * xyz, norm)
    to_tile = tf.constant((1., 0., 0.), dtype=axis_general_case.dtype)
    norm_flat = tf.reshape(norm, [-1])
    axis_shape = tf.shape(input=axis_general_case)
    axis_general_case_flat = tf.reshape(axis_general_case, [-1, 3])
    axis_small_norm_flat = tf.tile(to_tile, [tf.size(input=norm_flat)])
    axis_small_norm_flat = tf.reshape(axis_small_norm_flat,
                                      tf.shape(input=axis_general_case_flat))
    axis = tf.where(norm_flat < 1e-6, axis_small_norm_flat,
                    axis_general_case_flat)
    axis = tf.reshape(axis, axis_shape)
    return axis, angle


def from_rotation_matrix(rotation_matrix, name=None):
  """Convert a rotation vector to an axis angle representation.

  Note: in current version returned axis-angle representation is not unique
    for a given rotation matrix. Since a direct conversion would not really be
    faster, we first transoform rotation_matrix to quaternion, and finally
    perform the conversion from quaternion to axis-angle.

  Args:
    rotation_matrix: N-D tensor of shape `[?, ..., ?, 3, 3]`.
    name: A name for this op. Defaults to "axis_angle_from_rotation_matrix".

  Returns:
    Tuple of two N-D tensors of shape `[?, ..., ?, 3]` and `[?, ..., ?, 1]`. The
    first tensor represent the axis, and the second one the angle. The resulting
    axis-angle is normalized.

  Raises:
    ValueError: if the shape of `rotation_vector` is not supported.
  """
  with tf.compat.v1.name_scope(name, "axis_angle_from_rotation_matrix",
                               [rotation_matrix]):
    rotation_matrix = tf.convert_to_tensor(value=rotation_matrix)
    shape = rotation_matrix.shape.as_list()
    if shape[-2:] != [3, 3]:
      raise ValueError("'rotation_matrix' must have 3x3 dimensions.")

    rotation_matrix = rotation_matrix_3d.assert_rotation_matrix_normalized(
        rotation_matrix)
    quaternion = quaternion_lib.from_rotation_matrix(rotation_matrix)
    return from_quaternion(quaternion)


def from_rotation_vector(rotation_vector, name=None):
  """Convert a rotation vector to an axis angle representation.

  Args:
    rotation_vector: N-D tensor of shape `[?, ..., ?, 3]`.
    name: A name for this op. Defaults to "axis_angle_from_rotation_vector".

  Returns:
    Tuple of two N-D tensors of shape `[?, ..., ?, 3]` and `[?, ..., ?, 1]`.

  Raises:
    ValueError: if the shape of `rotation_vector` is not supported.
  """
  with tf.compat.v1.name_scope(name, "axis_angle_from_rotation_vector",
                               [rotation_vector]):
    rotation_vector = tf.convert_to_tensor(value=rotation_vector)
    shape = rotation_vector.shape.as_list()
    if shape[-1] != 3:
      raise ValueError("'rotation_vector' must have 3 dimensions.")

    angle = tf.norm(tensor=rotation_vector, axis=-1, keepdims=True)
    axis = safe_ops.safe_unsigned_div(rotation_vector, angle)
    return axis, angle


def inverse(axis, angle, name=None):
  """Computes the axis angle that is the inverse of the input axis angle.

  Args:
    axis: N-D tensor of shape `[?, ..., ?, 3]`.
    angle: N-D tensor of shape `[?, ..., ?, 1]`.
    name: A name for this op. Defaults to "axis_angle_inverse".

  Returns:
    Tuple of two N-D tensors of shape `[?, ..., ?, 3]` and `[?, ..., ?, 1]`.

  Raises:
    ValueError: if the shape of the arguments is not supported.
  """
  with tf.compat.v1.name_scope(name, "axis_angle_inverse", [axis, angle]):
    axis = tf.convert_to_tensor(value=axis)
    angle = tf.convert_to_tensor(value=angle)
    shape_axis = axis.shape.as_list()
    shape_angle = angle.shape.as_list()
    if shape_axis[-1] != 3:
      raise ValueError("'axis' must have 3 dimensions.")
    if shape_angle[-1] != 1:
      raise ValueError("'angle' must have 1 dimension.")

    axis = asserts.assert_normalized(axis)
    return axis, -angle


def is_normalized(axis, angle, atol=1e-3, name=None):
  """Determines if the axis-angle is normalized or not.

  Args:
    axis: N-D tensor of shape `[?, ..., ?, 3]`.
    angle: N-D tensor of shape `[?, ..., ?, 1]`.
    atol: Absolute tolerance parameter.
    name: A name for this op. Defaults to "axis_angle_is_normalized".

  Returns:
    A N-D tensor of shape `[?, ..., ?, 1]` where False indicates that the
    axis-angle is not normalized.
  """
  with tf.compat.v1.name_scope(name, "axis_angle_is_normalized", [axis, angle]):
    axis = tf.convert_to_tensor(value=axis)
    angle = tf.convert_to_tensor(value=angle)
    shape_axis = axis.shape.as_list()
    shape_angle = angle.shape.as_list()
    if shape_axis[-1] != 3:
      raise ValueError("'axis' must have 3 dimensions.")
    if shape_angle[-1] != 1:
      raise ValueError("'angle' must have 1 dimension.")
    norms = tf.norm(tensor=axis, axis=-1, keepdims=True)
    return tf.abs(norms - 1.) < atol


def rotate(point, axis, angle, name=None):
  """Rotate a point using an axis angle.

  Args:
    point: N-D tensor of shape `[?, ..., ?, 3]`.
    axis: N-D tensor of shape `[?, ..., ?, 3]`.
    angle: N-D tensor of shape `[?, ..., ?, 1]`.
    name: A name for this op. Defaults to "axis_angle_rotate".

  Returns:
    A N-D tensor of shape `[?, ..., ?, 3]`.

  Raises:
    ValueError: if `point`, `axis`, or `quaternion` are of different shape or if
    their respective shapeis not supported.
  """
  with tf.compat.v1.name_scope(name, "axis_angle_rotate", [point, axis, angle]):
    point = tf.convert_to_tensor(value=point)
    axis = tf.convert_to_tensor(value=axis)
    angle = tf.convert_to_tensor(value=angle)
    shape_point = point.shape.as_list()
    shape_axis = axis.shape.as_list()
    shape_angle = angle.shape.as_list()
    if shape_point[-1] != 3:
      raise ValueError("'point' must have 3 dimensions.")
    if shape_axis[-1] != 3:
      raise ValueError("'axis' must have 3 dimensions.")
    if shape_angle[-1] != 1:
      raise ValueError("'angle' must have 1 dimensions.")

    axis = asserts.assert_normalized(axis)
    cos_angle = tf.cos(angle)
    axis_dot_point = tf.reduce_sum(
        input_tensor=tf.multiply(axis, point), axis=-1, keepdims=True)
    res = point * cos_angle + tf.linalg.cross(
        axis, point) * tf.sin(angle) + axis * axis_dot_point * (1.0 - cos_angle)
    return res


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
