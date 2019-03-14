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
"""This module implements the Blinn-Phong specular reflectance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import math as m
import sys

import tensorflow as tf

from tensorflow_graphics.geometry import vector
from tensorflow_graphics.util import asserts


def brdf(direction_incoming_light,
         direction_outgoing_light,
         surface_normal,
         shininess,
         albedo,
         name=None):
  """Evaluates the specular brdf of the Blinn-Phong model.

  Note:
    This function returns a modified specular Blinn-Phong model that ensures
    energy conservation.

  Args:
    direction_incoming_light: N-D tensor of shape `[?, ..., ?, 3]`, L2
      normalized along the last dimension.
    direction_outgoing_light: N-D tensor of shape `[?, ..., ?, 3]`, L2
      normalized along the last dimension.
    surface_normal: N-D tensor of shape `[?, ..., ?, 3]`, L2 normalized along
      the last dimension.
    shininess: N-D tensor of shape `[?, ..., ?, 1]`.
    albedo: N-D tensor of shape `[?, ..., ?, 3]` with values in [0,1].
    name: A name for this op. Defaults to "blinn_phong_brdf".

  Returns:
      N-D tensor of shape `[?, ..., ?, 3]`.

  Raises:
    ValueError: if the shape of `direction_incoming_light`,
    `direction_outgoing_light`, `surface_normal`, `shininess` or `albedo` is not
    supported.
    InvalidArgumentError: if at least an element of `albedo` is outside of
    [0,1].
  """
  with tf.compat.v1.name_scope(name, "blinn_phong_brdf", [
      direction_incoming_light, direction_outgoing_light, surface_normal,
      shininess, albedo
  ]):
    # Conversion to tensors.
    direction_incoming_light = tf.convert_to_tensor(
        value=direction_incoming_light)
    direction_outgoing_light = tf.convert_to_tensor(
        value=direction_outgoing_light)
    surface_normal = tf.convert_to_tensor(value=surface_normal)
    shininess = tf.convert_to_tensor(value=shininess)
    albedo = tf.convert_to_tensor(value=albedo)
    # Check that the shape of inputs are supported.
    if direction_incoming_light.shape.as_list()[-1] != 3:
      raise ValueError("'direction_incoming_light' must have 3 dimensions.")
    if direction_outgoing_light.shape.as_list()[-1] != 3:
      raise ValueError("'direction_outgoing_light' must have 3 dimensions.")
    if surface_normal.shape.as_list()[-1] != 3:
      raise ValueError("'surface_normal' must have 3 dimensions.")
    if shininess.shape.as_list()[-1] != 1:
      raise ValueError("'shininess' must have 1 dimension.")
    if albedo.shape.as_list()[-1] != 3:
      raise ValueError("'albedo' must have 3 dimensions.")
    # Check that tensors contain appropriate data.
    direction_incoming_light = asserts.assert_normalized(
        direction_incoming_light)
    direction_outgoing_light = asserts.assert_normalized(
        direction_outgoing_light)
    surface_normal = asserts.assert_normalized(surface_normal)
    albedo = asserts.assert_all_in_range(albedo, 0.0, 1.0, open_bounds=False)
    # BRDF evaluation.
    difference_outgoing_incoming = (
        direction_outgoing_light - direction_incoming_light)
    difference_outgoing_incoming = tf.math.l2_normalize(
        difference_outgoing_incoming, axis=-1)
    cos_alpha = vector.dot(
        surface_normal, difference_outgoing_incoming, axis=-1)
    cos_alpha = tf.maximum(tf.zeros_like(cos_alpha), cos_alpha)
    return albedo * (shininess + 2.0) / (2.0 * m.pi) * tf.pow(
        cos_alpha, shininess)


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
