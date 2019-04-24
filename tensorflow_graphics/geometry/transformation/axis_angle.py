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
r"""This module implements axis-angle functionalities.

The axis-angle representation is defined as $$\theta\mathbf{a}$$, where
$$\mathbf{a}$$ is a unit vector indicating the direction of rotation and
$$\theta$$ is a scalar controlling the angle of rotation. It is important to
note that the axis-angle does not perform rotation by itself, but that it can be
used to rotate any given vector $$\mathbf{v} \in {\mathbb{R}^3}$$ into
a vector $$\mathbf{v}'$$ using the Rodrigues' rotation formula:

$$\mathbf{v}'=\mathbf{v}\cos(\theta)+(\mathbf{a}\times\mathbf{v})\sin(\theta)
+\mathbf{a}(\mathbf{a}\cdot\mathbf{v})(1-\cos(\theta)).$$

More details about the axis-angle formalism can be found on [this page.]
(https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation)

Note: Some of the functions defined in the module expect
a normalized axis $$\mathbf{a} = [x, y, z]^T$$ as inputs where
$$x^2 + y^2 + z^2 = 1$$.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.geometry.transformation import quaternion as quaternion_lib
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.math import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import safe_ops


def from_euler(angles, name=None):
  r"""Converts Euler angles to an axis-angle representation.

  Note:
    The conversion is performed by first converting to a quaternion
    representation, and then by converting the quaternion to an axis-angle.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three Euler angles. `[A1, ..., An, 0]` is the angle about
      `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians and
      `[A1, ..., An, 2]` is the angle about `z` in radians.
    name: A name for this op that defaults to "axis_angle_from_euler".

  Returns:
    A tuple of two tensors, respectively of shape `[A1, ..., An, 3]` and
    `[A1, ..., An, 1]`, where the first tensor represents the axis, and the
    second represents the angle. The resulting axis is a normalized vector.
  """
  with tf.compat.v1.name_scope(name, "axis_angle_from_euler", [angles]):
    quaternion = quaternion_lib.from_euler(angles)
    return from_quaternion(quaternion)


def from_euler_with_small_angles_approximation(angles, name=None):
  r"""Converts small Euler angles to an axis-angle representation.

  Under the small angle assumption, $$\sin(x)$$ and $$\cos(x)$$ can be
  approximated by their second order Taylor expansions, where
  $$\sin(x) \approx x$$ and $$\cos(x) \approx 1 - \frac{x^2}{2}$$.
  In the current implementation, the smallness of the angles is not verified.

  Note:
    The conversion is performed by first converting to a quaternion
    representation, and then by converting the quaternion to an axis-angle.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three small Euler angles. `[A1, ..., An, 0]` is the angle
      about `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians
      and `[A1, ..., An, 2]` is the angle about `z` in radians.
    name: A name for this op that defaults to
      "axis_angle_from_euler_with_small_angles_approximation".

  Returns:
    A tuple of two tensors, respectively of shape `[A1, ..., An, 3]` and
    `[A1, ..., An, 1]`, where the first tensor represents the axis, and the
    second represents the angle. The resulting axis is a normalized vector.
  """
  with tf.compat.v1.name_scope(
      name, "axis_angle_from_euler_with_small_angles_approximation", [angles]):
    quaternion = quaternion_lib.from_euler_with_small_angles_approximation(
        angles)
    return from_quaternion(quaternion)


def from_quaternion(quaternion, name=None):
  """Converts a quaternion to an axis-angle representation.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to "axis_angle_from_quaternion".

  Returns:
    Tuple of two tensors of shape `[A1, ..., An, 3]` and `[A1, ..., An, 1]`,
    where the first tensor represents the axis, and the second represents the
    angle. The resulting axis is a normalized vector.

  Raises:
    ValueError: If the shape of `quaternion` is not supported.
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
  """Converts a rotation matrix to an axis-angle representation.

  Note:
    In the current version the returned axis-angle representation is not unique
    for a given rotation matrix. Since a direct conversion would not really be
    faster, we first transform the rotation matrix to a quaternion, and finally
    perform the conversion from that quaternion to the corresponding axis-angle
    representation.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    rotation_matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
      dimensions represent a rotation matrix.
    name: A name for this op that defaults to "axis_angle_from_rotation_matrix".

  Returns:
    A tuple of two tensors, respectively of shape `[A1, ..., An, 3]` and
    `[A1, ..., An, 1]`, where the first tensor represents the axis, and the
    second represents the angle. The resulting axis is a normalized vector.

  Raises:
    ValueError: If the shape of `rotation_matrix` is not supported.
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
  r"""Converts a rotation vector to an axis-angle representation.

  A rotation vector is a vector $$r \in \mathbb{R}^3$$ where
  $$\frac{r}{\|r\|_2} = \mathbf{a}$$ is a unit vector indicating the axis of
  rotation and $$\|r\|_2 = \theta$$ is the angle.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    rotation_vector: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents a rotation vector.
    name: A name for this op that defaults to "axis_angle_from_rotation_vector".

  Returns:
    A tuple of two tensors, respectively of shape `[A1, ..., An, 3]` and
    `[A1, ..., An, 1]`, where the first tensor represents the axis, and the
    second represents the angle. The resulting axis is a normalized vector.

  Raises:
    ValueError: If the shape of `rotation_vector` is not supported.
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
  """Computes the axis-angle that is the inverse of the input axis-angle.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]` where the last dimension
      represents an angle.
    name: A name for this op that defaults to "axis_angle_inverse".

  Returns:
    A tuple of two tensors, respectively of shape `[A1, ..., An, 3]` and
    `[A1, ..., An, 1]`, where the first tensor represents the axis, and the
    second represents the angle. The resulting axis is a normalized vector.

  Raises:
    ValueError: If the shape of `axis` or `angle` is not supported.
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

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]` where the last dimension
      represents an angle.
    atol: The absolute tolerance parameter.
    name: A name for this op that defaults to "axis_angle_is_normalized".

  Returns:
    A tensor of shape `[A1, ..., An, 1]`, where False indicates that the axis is
    not normalized.
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
  r"""Rotates a 3d point using an axis-angle by applying the Rodrigues' formula.

  Rotates a vector $$\mathbf{v} \in {\mathbb{R}^3}$$ into a vector
  $$\mathbf{v}' \in {\mathbb{R}^3}$$ using the Rodrigues' rotation formula:

  $$\mathbf{v}'=\mathbf{v}\cos(\theta)+(\mathbf{a}\times\mathbf{v})\sin(\theta)
  +\mathbf{a}(\mathbf{a}\cdot\mathbf{v})(1-\cos(\theta)).$$

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    point: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point to rotate.
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents an angle.
    name: A name for this op that defaults to "axis_angle_rotate".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    a 3d point.

  Raises:
    ValueError: If `point`, `axis`, or `angle` are of different shape or if
    their respective shape is not supported.
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
    axis_dot_point = vector.dot(axis, point)
    res = point * cos_angle + vector.cross(
        axis, point) * tf.sin(angle) + axis * axis_dot_point * (1.0 - cos_angle)
    return res


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
