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
"""This contains routines shared for 2d and 3d rotation matrices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import tensorflow as tf


def is_normalized(rotation_matrix, atol=1e-3, name=None):
  """Determines if a matrix in K-dimensions is a valid rotation matrix.

  Args:
    rotation_matrix: N-D tensor of shape `[?, ..., ?, K, K]`.
    atol: Absolute tolerance parameter.
    name: A name for this op. Defaults to
      "rotation_matrix_common_is_normalized".

  Returns:
    A N-D tensor of shape `[?, ..., ?, 1]` where False indicates that the
    input is not a valid rotation matrix.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_common_is_normalized",
                               [rotation_matrix]):
    rotation_matrix = tf.convert_to_tensor(value=rotation_matrix)
    shape = rotation_matrix.shape.as_list()
    shape_length = len(shape)
    if shape_length < 2 or shape[-1] != shape[-2]:
      raise ValueError("'rotation_matrix' must have KxK dimensions.")
    # Computes the determinants
    distance_to_unit_determinant = tf.abs(tf.linalg.det(rotation_matrix) - 1.)
    # Computes how far the product of the transposed rotation matrix with itself
    # is from the identity matrix.
    permutation = list(
        range(shape_length - 2)) + [shape_length - 1, shape_length - 2]
    identity = tf.eye(shape[-1], dtype=rotation_matrix.dtype)
    difference_to_identity = tf.matmul(
        tf.transpose(a=rotation_matrix, perm=permutation),
        rotation_matrix) - identity
    norm_diff = tf.norm(tensor=difference_to_identity, axis=(-2, -1))
    # Computes the mask of entries that satisfies all conditions.
    mask = tf.logical_and(distance_to_unit_determinant < atol, norm_diff < atol)
    output = tf.where(mask,
                      tf.ones_like(distance_to_unit_determinant, dtype=bool),
                      tf.zeros_like(distance_to_unit_determinant, dtype=bool))
    return tf.expand_dims(output, axis=-1)


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
