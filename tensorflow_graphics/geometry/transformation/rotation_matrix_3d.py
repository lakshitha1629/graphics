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
"""This module implements TensorFlow 3d rotation matrix utility functions.

More details rotation matrices can be found on [this page.]
(https://en.wikipedia.org/wiki/Rotation_matrix)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import rotation_matrix_common
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import tfg_flags

FLAGS = flags.FLAGS


def _build_matrix_from_sines_and_cosines(sin_angles, cos_angles):
  """Builds a rotation matrix from sines and cosines of Euler angles.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    sin_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the sine of the Euler angles.
    cos_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the cosine of the Euler angles.

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.
  """
  sin_angles.shape.assert_is_compatible_with(cos_angles.shape)
  sx, sy, sz = tf.unstack(sin_angles, axis=-1)
  cx, cy, cz = tf.unstack(cos_angles, axis=-1)
  m00 = cy * cz
  m01 = (sx * sy * cz) - (cx * sz)
  m02 = (cx * sy * cz) + (sx * sz)
  m10 = cy * sz
  m11 = (sx * sy * sz) + (cx * cz)
  m12 = (cx * sy * sz) - (sx * cz)
  m20 = -sy
  m21 = sx * cy
  m22 = cx * cy
  # pyformat: disable
  matrix = tf.stack((m00, m01, m02,
                     m10, m11, m12,
                     m20, m21, m22),
                    axis=-1)
  # pyformat: enable
  output_shape = tf.concat((tf.shape(input=sin_angles)[:-1], (3, 3)), axis=-1)
  return tf.reshape(matrix, shape=output_shape)


def assert_rotation_matrix_normalized(matrix, eps=None, name=None):
  """Checks whether a matrix is a rotation matrix.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
      dimensions represent a 3d rotation matrix.
    eps: The absolute tolerance parameter.
    name: A name for this op that defaults to
      'assert_rotation_matrix_normalized'.

  Returns:
    The input matrix, with dependence on the assertion operator in the graph.

  Raises:
    tf.errors.InvalidArgumentError: If rotation_matrix_3d is not normalized.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return matrix

  with tf.compat.v1.name_scope(name, "assert_rotation_matrix_normalized",
                               [matrix, eps]):
    matrix = tf.convert_to_tensor(value=matrix)
    shape = matrix.shape.as_list()
    shape_length = len(shape)
    if shape_length < 2 or shape[-1] != 3 or shape[-2] != 3:
      raise ValueError("'rotation_matrix_3d' must have 3x3 dimensions.")
    is_matrix_normalized = is_valid(matrix)
    with tf.control_dependencies([
        tf.compat.v1.assert_equal(
            is_matrix_normalized,
            tf.ones_like(is_matrix_normalized, dtype=tf.bool))
    ]):
      return tf.identity(matrix)


def from_axis_angle(axis, angle, name=None):
  """Convert an axis-angle representation to a rotation matrix.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents a normalized axis.
    name: A name for this op that defaults to
      "rotation_matrix_3d_from_axis_angle".

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represents a 3d rotation matrix.

  Raises:
    ValueError: If the shape of `axis` or `angle` is not supported.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_3d_from_axis_angle",
                               [axis, angle]):
    axis = tf.convert_to_tensor(value=axis)
    angle = tf.convert_to_tensor(value=angle)
    shape_axis = axis.shape.as_list()
    shape_angle = angle.shape.as_list()
    if shape_axis[-1] != 3:
      raise ValueError("'axis' must have 3 dimensions.")
    if shape_angle[-1] != 1:
      raise ValueError("'angle' must have 1 dimension.")

    axis = asserts.assert_normalized(axis)
    sin_axis = tf.sin(angle) * axis
    c = tf.cos(angle)
    cos1_axis = (1.0 - c) * axis
    _, axis_y, axis_z = tf.unstack(axis, axis=-1)
    cos1_axis_x, cos1_axis_y, _ = tf.unstack(cos1_axis, axis=-1)
    sin_axis_x, sin_axis_y, sin_axis_z = tf.unstack(sin_axis, axis=-1)
    tmp = cos1_axis_x * axis_y
    m01 = tmp - sin_axis_z
    m10 = tmp + sin_axis_z
    tmp = cos1_axis_x * axis_z
    m02 = tmp + sin_axis_y
    m20 = tmp - sin_axis_y
    tmp = cos1_axis_y * axis_z
    m12 = tmp - sin_axis_x
    m21 = tmp + sin_axis_x
    diag = cos1_axis * axis + c
    diag_x, diag_y, diag_z = tf.unstack(diag, axis=-1)
    # pyformat: disable
    matrix = tf.stack((diag_x, m01, m02,
                       m10, diag_y, m12,
                       m20, m21, diag_z),
                      axis=-1)
    # pyformat: enable
    output_shape = tf.concat((tf.shape(input=axis)[:-1], (3, 3)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


def from_euler(angles, name=None):
  r"""Convert an Euler angle representation to a rotation matrix.

  The resulting matrix is $$\mathbf{R} = \mathbf{R}_z\mathbf{R}_y\mathbf{R}_x$$.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three Euler angles. `[A1, ..., An, 0]` is the angle about
      `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians and
      `[A1, ..., An, 2]` is the angle about `z` in radians.
    name: A name for this op that defaults to "rotation_matrix_3d_from_euler".

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.

  Raises:
    ValueError: If the shape of `angles` is not supported.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_3d_from_euler", [angles]):
    angles = tf.convert_to_tensor(value=angles)
    shape_angles = angles.shape.as_list()
    if shape_angles[-1] != 3:
      raise ValueError("'angles' must have 3 dimensions.")

    sin_angles = tf.sin(angles)
    cos_angles = tf.cos(angles)
    return _build_matrix_from_sines_and_cosines(sin_angles, cos_angles)


def from_euler_with_small_angles_approximation(angles, name=None):
  r"""Convert an Euler angle representation to a rotation matrix.

  The resulting matrix is $$\mathbf{R} = \mathbf{R}_z\mathbf{R}_y\mathbf{R}_x$$.
  Under the small angle assumption, $$\sin(x)$$ and $$\cos(x)$$ can be
  approximated by their second order Taylor expansions, where
  $$\sin(x) \approx x$$ and $$\cos(x) \approx 1 - \frac{x^2}{2}$$.
  In the current implementation, the smallness of the angles is not verified.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three small Euler angles. `[A1, ..., An, 0]` is the angle
      about `x` in radians, `[A1, ..., An, 1]` is the angle about `y` in radians
      and `[A1, ..., An, 2]` is the angle about `z` in radians.
    name: A name for this op that defaults to "rotation_matrix_3d_from_euler".

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.

  Raises:
    ValueError: If the shape of `angles` is not supported.
  """
  with tf.compat.v1.name_scope(
      name, "rotation_matrix_3d_from_euler_with_small_angles", [angles]):
    angles = tf.convert_to_tensor(value=angles)
    shape_angles = angles.shape.as_list()
    if shape_angles[-1] != 3:
      raise ValueError("'angles' must have 3 dimensions.")

    sin_angles = angles
    cos_angles = 1.0 - 0.5 * angles * angles
    return _build_matrix_from_sines_and_cosines(sin_angles, cos_angles)


def from_quaternion(quaternion, name=None):
  """Convert a quaternion to a rotation matrix.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to
      "rotation_matrix_3d_from_quaternion".

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.

  Raises:
    ValueError: If the shape of `quaternion` is not supported.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_3d_from_quaternion",
                               [quaternion]):
    quaternion = tf.convert_to_tensor(value=quaternion)
    shape = quaternion.shape.as_list()
    if shape[-1] != 4:
      raise ValueError("'quaternion' must have 4 dimensions.")

    quaternion = asserts.assert_normalized(quaternion)
    x, y, z, w = tf.unstack(quaternion, axis=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    # pyformat: disable
    matrix = tf.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                       txy + twz, 1.0 - (txx + tzz), tyz - twx,
                       txz - twy, tyz + twx, 1.0 - (txx + tyy)),
                      axis=-1)
    # pyformat: enable
    output_shape = tf.concat((tf.shape(input=quaternion)[:-1], (3, 3)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


def inverse(matrix, name=None):
  """Computes the inverse of a 3D rotation matrix.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
      dimensions represent a 3d rotation matrix.
    name: A name for this op that defaults to "rotation_matrix_3d_inverse".

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.

  Raises:
    ValueError: If the shape of `matrix` is not supported.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_3d_inverse", [matrix]):
    matrix = tf.convert_to_tensor(value=matrix)
    shape_matrix = matrix.shape.as_list()
    if shape_matrix[-2:] != [3, 3]:
      raise ValueError("'matrix' must have 3x3 dimensions.")

    matrix = assert_rotation_matrix_normalized(matrix)
    input_len = len(shape_matrix)
    perm = list(range(input_len - 2)) + [input_len - 1, input_len - 2]
    return tf.transpose(a=matrix, perm=perm)


def is_valid(matrix, atol=1e-3, name=None):
  """Determines if a matrix is a valid rotation matrix.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    matrix: A tensor of shape `[A1, ..., An, 3,3]`, where the last two
      dimensions represent a matrix.
    atol: Absolute tolerance parameter.
    name: A name for this op that defaults to "rotation_matrix_3d_is_valid".

  Returns:
    A tensor of type `bool` and shape `[A1, ..., An, 1]` where False indicates
    that the input is not a valid rotation matrix.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_3d_is_valid", [matrix]):
    matrix = tf.convert_to_tensor(value=matrix)
    shape = matrix.shape.as_list()
    shape_length = len(shape)
    if shape_length < 2 or shape[-1] != 3 or shape[-2] != 3:
      raise ValueError("'matrix' must have 3x3 dimensions.")
    return rotation_matrix_common.is_valid(matrix, atol)


def rotate(point, matrix, name=None):
  """Rotate a point using a rotation matrix 3d.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    point: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point.
    matrix: A tensor of shape `[A1, ..., An, 3,3]`, where the last dimension
      represents a 3d rotation matrix.
    name: A name for this op that defaults to "rotation_matrix_3d_rotate".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    a 3d point.

  Raises:
    ValueError: If the shape of `point` or `rotation_matrix_3d` is not
    supported.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_3d_rotate",
                               [point, matrix]):
    point = tf.convert_to_tensor(value=point)
    matrix = tf.convert_to_tensor(value=matrix)
    shape_point = point.shape.as_list()
    shape_matrix = matrix.shape.as_list()
    if shape_point[-1] != 3:
      raise ValueError("'point' must have 3 dimensions.")
    if shape_matrix[-2:] != [3, 3]:
      raise ValueError("'matrix' must have 3x3 dimensions.")

    matrix = assert_rotation_matrix_normalized(matrix)
    point = tf.expand_dims(point, axis=-1)
    rotated_point = tf.matmul(matrix, point)
    return tf.squeeze(rotated_point, axis=-1)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
