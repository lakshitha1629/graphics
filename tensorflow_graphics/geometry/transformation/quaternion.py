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
"""This module implements TensorFlow quaternion utility functions.

A quaternion is written as $$q =  xi + yj + zk + w$$, where $$i,j,k$$ forms the
three bases of the imaginary part. The functions implemented in this file
use the Hamilton convention where $$i^2 = j^2 = k^2 = ijk = -1$$. A quaternion
is stored in a 4-D vector $$[x, y, z, w]^T$$.

More details about Hamiltonian quaternions can be found on [this page.]
(https://en.wikipedia.org/wiki/Quaternion)

Note: Some of the functions expect normalized quaternions as inputs where
$$x^2 + y^2 + z^2 + w^2 = 1$$.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.math import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import safe_ops


def _build_quaternion_from_sines_and_cosines(sin_half_angles, cos_half_angles):
  """Builds a quaternion from sines and cosines of half Euler angles.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    sin_half_angles: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the sine of half Euler angles.
    cos_half_angles: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the cosine of half Euler angles.

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a quaternion.
  """
  c1, c2, c3 = tf.unstack(cos_half_angles, axis=-1)
  s1, s2, s3 = tf.unstack(sin_half_angles, axis=-1)
  w = c1 * c2 * c3 + s1 * s2 * s3
  x = -c1 * s2 * s3 + s1 * c2 * c3
  y = c1 * s2 * c3 + s1 * c2 * s3
  z = -s1 * s2 * c3 + c1 * c2 * s3
  return tf.stack((x, y, z, w), axis=-1)


def between_two_vectors_3d(vector1, vector2, name=None):
  """Computes quaternion over the shortest arc between two vectors.

  Result quaternion describes shortest geodesic rotation from
  vector1 to vector2.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vector1: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the first vector.
    vector2: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the second vector.
    name: A name for this op that defaults to
      "quaternion_between_two_vectors_3d".

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.

  Raises:
    ValueError: If the shape of `vector1` or `vector2` is not supported.
  """
  with tf.compat.v1.name_scope(name, "quaternion_between_two_vectors_3d",
                               [vector1, vector2]):
    vector1 = tf.convert_to_tensor(value=vector1)
    vector2 = tf.convert_to_tensor(value=vector2)
    shape1 = vector1.shape.as_list()
    shape2 = vector2.shape.as_list()
    if shape1[-1] != 3:
      raise ValueError("'vector1' must have 3 dimensions.")
    if shape2[-1] != 3:
      raise ValueError("'vector2' must have 3 dimensions.")
    # Make sure we deal with unit vectors.
    vector1 = tf.nn.l2_normalize(vector1, axis=-1)
    vector2 = tf.nn.l2_normalize(vector2, axis=-1)
    axis = vector.cross(vector1, vector2)
    cos_theta = vector.dot(vector1, vector2)
    rot = tf.concat((axis, 1. + cos_theta), axis=-1)
    return tf.nn.l2_normalize(rot, axis=-1)


def conjugate(quaternion, name=None):
  """Computes the conjugate of a quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to "quaternion_conjugate".

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.

  Raises:
    ValueError: If the shape of `quaternion` is not supported.
  """
  with tf.compat.v1.name_scope(name, "quaternion_conjugate", [quaternion]):
    quaternion = tf.convert_to_tensor(value=quaternion)
    shape = quaternion.shape.as_list()
    if shape[-1] != 4:
      raise ValueError("'quaternion' must have 4 dimensions.")

    quaternion = asserts.assert_normalized(quaternion)
    xyz, w = tf.split(quaternion, (3, 1), axis=-1)
    return tf.concat((-xyz, w), axis=-1)


def from_axis_angle(axis, angle, name=None):
  """Converts an axis-angle representation to a quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents an angle.
    name: A name for this op that defaults to "quaternion_from_axis_angle".

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.

  Raises:
    ValueError: If the shape of `axis` or `angle` is not supported.
  """
  with tf.compat.v1.name_scope(name, "quaternion_from_axis_angle",
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
    half_angle = 0.5 * angle
    w = tf.cos(half_angle)
    xyz = tf.sin(half_angle) * axis
    return tf.concat((xyz, w), axis=-1)


def from_euler(angles, name=None):
  """Converts an Euler angle representation to a quaternion.

  Note:
    Uses the z-y-x rotation convention (Tait-Bryan angles).

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three Euler angles. `[A1, ..., An, 0]` is the angle about
      `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians and
      `[A1, ..., An, 2]` is the angle about `z` in radians.
    name: A name for this op that defaults to "quaternion_from_euler".

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.

  Raises:
    ValueError: If the shape of `angles` is not supported.
  """
  with tf.compat.v1.name_scope(name, "quaternion_from_euler", [angles]):
    angles = tf.convert_to_tensor(value=angles)
    angles_shape = angles.shape.as_list()
    if angles_shape[-1] != 3:
      raise ValueError("'angles' must have 3 dimensions.")
    half_angles = angles / 2.0
    cos_half_angles = tf.cos(half_angles)
    sin_half_angles = tf.sin(half_angles)
    return _build_quaternion_from_sines_and_cosines(sin_half_angles,
                                                    cos_half_angles)


def from_euler_with_small_angles_approximation(angles, name=None):
  r"""Converts small Euler angles to quaternions.

  Under the small angle assumption, $$\sin(x)$$ and $$\cos(x)$$ can be
  approximated by their second order Taylor expansions, where
  $$\sin(x) \approx x$$ and $$\cos(x) \approx 1 - \frac{x^2}{2}$$.
  In the current implementation, the smallness of the angles is not verified.

  Note:
    Uses the z-y-x rotation convention (Tait-Bryan angles).

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three small Euler angles. `[A1, ..., An, 0]` is the angle
      about `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians
      and `[A1, ..., An, 2]` is the angle about `z` in radians.
    name: A name for this op that defaults to "quaternion_from_euler".

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.

  Raises:
    ValueError: If the shape of `angles` is not supported.
  """
  with tf.compat.v1.name_scope(name, "quaternion_from_euler", [angles]):
    angles = tf.convert_to_tensor(value=angles)
    angles_shape = angles.shape.as_list()
    if angles_shape[-1] != 3:
      raise ValueError("'angles' must have 3 dimensions.")
    half_angles = angles / 2.0
    cos_half_angles = 1.0 - 0.5 * half_angles * half_angles
    sin_half_angles = half_angles
    quaternion = _build_quaternion_from_sines_and_cosines(
        sin_half_angles, cos_half_angles)
    # We need to normalize the quaternion due to the small angle approximation.
    return tf.nn.l2_normalize(quaternion, axis=-1)


def from_rotation_matrix(rotation_matrix, name=None):
  """Converts a rotation matrix representation to a quaternion.

  Warning:
    This function is not smooth everywhere.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    rotation_matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
      dimensions represent a rotation matrix.
    name: A name for this op that defaults to "quaternion_from_rotation_matrix".

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.

  Raises:
    ValueError: If the shape of `rotation_matrix` is not supported.
  """
  with tf.compat.v1.name_scope(name, "quaternion_from_rotation_matrix",
                               [rotation_matrix]):
    rotation_matrix = tf.convert_to_tensor(value=rotation_matrix)
    shape = rotation_matrix.shape.as_list()
    if shape[-2:] != [3, 3]:
      raise ValueError("'rotation_matrix' must have 3x3 dimensions.")

    rotation_matrix = rotation_matrix_3d.assert_rotation_matrix_normalized(
        rotation_matrix)

    tr = tf.linalg.trace(rotation_matrix)
    m = rotation_matrix
    eps_addition = asserts.select_eps_for_addition(m.dtype)

    def tr_positive(m):
      s = tf.sqrt(tr + 1.0 + eps_addition) * 2.  # s=4*qw
      qw = 0.25 * s
      qx = safe_ops.safe_unsigned_div(m[..., 2, 1] - m[..., 1, 2], s)
      qy = safe_ops.safe_unsigned_div(m[..., 0, 2] - m[..., 2, 0], s)
      qz = safe_ops.safe_unsigned_div(m[..., 1, 0] - m[..., 0, 1], s)
      return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_1(m):
      s = tf.sqrt(1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2] +
                  eps_addition) * 2.  # s=4*qx
      qw = safe_ops.safe_unsigned_div(m[..., 2, 1] - m[..., 1, 2], s)
      qx = 0.25 * s
      qy = safe_ops.safe_unsigned_div(m[..., 0, 1] + m[..., 1, 0], s)
      qz = safe_ops.safe_unsigned_div(m[..., 0, 2] + m[..., 2, 0], s)
      return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_2(m):
      s = tf.sqrt(1.0 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2] +
                  eps_addition) * 2.  # s=4*qy
      qw = safe_ops.safe_unsigned_div(m[..., 0, 2] - m[..., 2, 0], s)
      qx = safe_ops.safe_unsigned_div(m[..., 0, 1] + m[..., 1, 0], s)
      qy = 0.25 * s
      qz = safe_ops.safe_unsigned_div(m[..., 1, 2] + m[..., 2, 1], s)
      return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_3(m):
      s = tf.sqrt(1.0 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1] +
                  eps_addition) * 2.  # s=4*q
      qw = safe_ops.safe_unsigned_div(m[..., 1, 0] - m[..., 0, 1], s)
      qx = safe_ops.safe_unsigned_div(m[..., 0, 2] + m[..., 2, 0], s)
      qy = safe_ops.safe_unsigned_div(m[..., 1, 2] + m[..., 2, 1], s)
      qz = 0.25 * s
      return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_idx(cond):
      cond = tf.expand_dims(cond, -1)
      cond = tf.tile(cond, [1] * (len(shape) - 2) + [4])
      return cond

    where_2 = tf.where(
        cond_idx(m[..., 1, 1] > m[..., 2, 2]), cond_2(m), cond_3(m))
    where_1 = tf.where(
        cond_idx((m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])),
        cond_1(m), where_2)
    quat = tf.where(cond_idx(tr > 0), tr_positive(m), where_1)
    return quat


def inverse(quaternion, name=None):
  """Computes the inverse of a quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to "quaternion_inverse".

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.

  Raises:
    ValueError: If the shape of `quaternion` is not supported.
  """
  with tf.compat.v1.name_scope(name, "quaternion_inverse", [quaternion]):
    quaternion = tf.convert_to_tensor(value=quaternion)
    shape = quaternion.shape.as_list()
    if shape[-1] != 4:
      raise ValueError("'quaternion' must have 4 dimensions.")

    quaternion = asserts.assert_normalized(quaternion)
    qq = quaternion * quaternion
    squared_norm = tf.reduce_sum(input_tensor=qq, axis=-1, keepdims=True)
    return safe_ops.safe_unsigned_div(conjugate(quaternion), squared_norm)


def is_normalized(quaternion, atol=1e-3, name=None):
  """Determines if quaternion is normalized quaternion or not.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a quaternion.
    atol: The absolute tolerance parameter.
    name: A name for this op that defaults to "quaternion_is_normalized".

  Returns:
    A tensor of type `bool` and shape `[A1, ..., An, 1]`, where False indicates
    that the quaternion is not normalized.

  Raises:
    ValueError: If the shape of `quaternion` is not supported.
  """
  with tf.compat.v1.name_scope(name, "quaternion_is_normalized", [quaternion]):
    quaternion = tf.convert_to_tensor(value=quaternion)
    shape = quaternion.shape.as_list()
    if shape[-1] != 4:
      raise ValueError("'quaternion' must have 4 dimensions.")
    norms = tf.norm(tensor=quaternion, axis=-1, keepdims=True)
    return tf.where(
        tf.abs(norms - 1.) < atol, tf.ones_like(norms, dtype=bool),
        tf.zeros_like(norms, dtype=bool))


def normalize(quaternion, eps=1e-12, name=None):
  """Normalizes a quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a quaternion.
    eps: A lower bound value for the norm that defaults to 1e-12.
    name: A name for this op that defaults to "quaternion_normalize".

  Returns:
    A N-D tensor of shape `[?, ..., ?, 1]` where the quaternion elements have
    been normalized.

  Raises:
    ValueError: If the shape of `quaternion` is not supported.
  """
  with tf.compat.v1.name_scope(name, "quaternion_normalize", [quaternion]):
    quaternion = tf.convert_to_tensor(value=quaternion)
    shape = quaternion.shape.as_list()
    if shape[-1] != 4:
      raise ValueError("'quaternion' must have 4 dimensions.")
    return tf.math.l2_normalize(quaternion, axis=-1, epsilon=eps)


def multiply(quaternion1, quaternion2, name=None):
  """Multiplies two quaternions.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion1:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    quaternion2:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to "quaternion_multiply".

  Returns:
    A tensor of shape `[A1, ..., An, 4]` representing normalized quaternions.

  Raises:
    ValueError: If the shape of `quaternion1` or `quaternion2` is not supported.
  """
  with tf.compat.v1.name_scope(name, "quaternion_multiply",
                               [quaternion1, quaternion2]):
    quaternion1 = tf.convert_to_tensor(value=quaternion1)
    quaternion2 = tf.convert_to_tensor(value=quaternion2)
    shape1 = quaternion1.shape.as_list()
    shape2 = quaternion2.shape.as_list()
    if shape1[-1] != 4:
      raise ValueError("'quaternion1' must have 4 dimensions.")
    if shape2[-1] != 4:
      raise ValueError("'quaternion2' must have 4 dimensions.")

    x1, y1, z1, w1 = tf.unstack(quaternion1, axis=-1)
    x2, y2, z2, w2 = tf.unstack(quaternion2, axis=-1)
    x = x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
    y = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
    z = x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2
    w = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
    return tf.stack((x, y, z, w), axis=-1)


def normalized_random_uniform(shape, name=None):
  """Random normalized quaternion following a uniform distribution law on SO(3).

  Args:
    shape: A list representing the shape of the output tensor.
    name: A name for this op that defaults to
      "quaternion_normalized_random_uniform".

  Returns:
  """
  with tf.compat.v1.name_scope(name, "quaternion_normalized_random_uniform"):
    u1 = tf.random.uniform(shape, minval=0.0, maxval=1.0)
    u2 = tf.random.uniform(shape, minval=0.0, maxval=2.0 * math.pi)
    u3 = tf.random.uniform(shape, minval=0.0, maxval=2.0 * math.pi)
    a = tf.sqrt(1.0 - u1)
    b = tf.sqrt(u1)
    # pyformat: disable
    return tf.stack((a * tf.sin(u2),
                     a * tf.cos(u2),
                     b * tf.sin(u3),
                     b * tf.cos(u3)),
                    axis=-1)
    # pyformat: enable


def normalized_random_uniform_initializer():
  """Random unit quaternion initializer."""

  def _initializer(shape, dtype=tf.float32, partition_info=None):
    """Generate a random normalized quaternion.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      shape: A list representing the shape of the output. The last entry of the
        list must be `4`.
      dtype: type of the output (tf.float32 is the only type supported).
      partition_info: how the variable is partitioned (not used).

    Returns:
      A tensor of shape `[A1, ..., An, 4]` representing normalized quaternions.

    Raises:
      ValueError: If `shape` or `dtype` are not supported.
    """
    del partition_info  # unused
    if dtype != tf.float32:
      raise ValueError("'dtype' must be tf.float32.")
    if shape[-1] != 4:
      raise ValueError("Last dimension of 'shape' must be 4.")
    return normalized_random_uniform(shape[:-1])

  return _initializer


def rotate(point, quaternion, name=None):
  """Rotates a point using a quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    point: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point.
    quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to "quaternion_rotate".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents a
    3d point.

  Raises:
    ValueError: If the shape of `point` or `quaternion` is not supported.
  """
  with tf.compat.v1.name_scope(name, "quaternion_rotate", [point, quaternion]):
    point = tf.convert_to_tensor(value=point)
    quaternion = tf.convert_to_tensor(value=quaternion)
    shape_point = point.shape.as_list()
    shape_quaternion = quaternion.shape.as_list()
    if shape_point[-1] != 3:
      raise ValueError("'point' must have 3 dimensions.")
    if shape_quaternion[-1] != 4:
      raise ValueError("'quaternion' must have 4 dimensions.")

    quaternion = asserts.assert_normalized(quaternion)
    padding = [[0, 0] for _ in range(len(shape_point))]
    padding[-1][-1] = 1
    point = tf.pad(tensor=point, paddings=padding, mode="CONSTANT")
    point = multiply(quaternion, point)
    point = multiply(point, conjugate(quaternion))
    xyz, _ = tf.split(point, (3, 1), axis=-1)
    return xyz


def relative_angle(quaternion1, quaternion2, name=None):
  r"""Computes the unsigned relative rotation angle between 2 unit quaternions.

  Given two normalized quanternions $$\mathbf{q}_1$$ and $$\mathbf{q}_2$$, the
  relative angle is computed as
  $$\theta = 2\arccos(\mathbf{q}_1^T\mathbf{q}_2)$$.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion1: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    quaternion2: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to "quaternion_relative_angle".

  Returns:
    A tensor of shape `[A1, ..., An, 1]` where the last dimension represents
    rotation angles in the range [0.0, pi].

  Raises:
    ValueError: If the shape of `quaternion1` or `quaternion2` is not supported.
  """
  with (tf.compat.v1.name_scope(name, "quaternion_relative_angle",
                                [quaternion1, quaternion2])):
    quaternion1 = tf.convert_to_tensor(value=quaternion1)
    quaternion2 = tf.convert_to_tensor(value=quaternion2)
    shape1 = quaternion1.shape.as_list()
    shape2 = quaternion2.shape.as_list()
    if shape1[-1] != 4:
      raise ValueError("'quaternion1' must have 4 dimensions.")
    if shape2[-1] != 4:
      raise ValueError("'quaternion2' must have 4 dimensions.")

    # Ensure we deal with unit quaternions.
    quaternion1 = asserts.assert_normalized(quaternion1)
    quaternion2 = asserts.assert_normalized(quaternion2)
    dot_product = vector.dot(quaternion1, quaternion2, keepdims=False)
    # Ensure dot product is in range [-1. 1].
    eps_dot_prod = 4.0 * asserts.select_eps_for_addition(dot_product.dtype)
    dot_product = safe_ops.safe_shrink(
        dot_product, -1.0, 1.0, False, eps=eps_dot_prod)
    angle = 2 * tf.acos(tf.abs(dot_product))

    return angle


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
