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
"""This module implements phong specular reflectance.

For a derivation of the normalization factor ensuring energy conservation, we
refer the interested reader to
Eric P. Lafortune, and Yves D. Willems.
"Using the modified phong reflectance model for physically based rendering".
1994.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from tensorflow_graphics.math import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api


def _brdf_normalization_factor(shininess):
  """Returns the normalization factor needed to ensure energy conservation."""
  return (shininess + 2.0) / (2.0 * math.pi)


def brdf(direction_incoming_light,
         direction_outgoing_light,
         surface_normal,
         shininess,
         albedo,
         brdf_normalization=True,
         name=None):
  """Evaluates the specular brdf of the Phong model.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Note:
    The gradient of this function is not smooth when the dot product of the
    normal with any light is 0.0.

  Args:
    direction_incoming_light: A tensor of shape `[A1, ..., An, 3]`, where the
      last dimension represents a normalized incoming light vector.
    direction_outgoing_light: A tensor of shape `[A1, ..., An, 3]`, where the
      last dimension represents a normalized outgoing light vector.
    surface_normal: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents a normalized surface normal.
    shininess: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents a shininess coefficient.
    albedo: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents albedo with values in [0,1].
    brdf_normalization: A `bool` indicating whether normalization should be
      applied to enforce the energy conservation property of BRDFs. Note that
      `brdf_normalization` must be set to False in order to use the original
      Blinn specular model.
    name: A name for this op. Defaults to "phong_brdf".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
      the amount of light reflected in the outgoing light direction.

  Raises:
    ValueError: if the shape of `direction_incoming_light`,
    `direction_outgoing_light`, `surface_normal`, `shininess` or `albedo` is not
    supported.
    InvalidArgumentError: if at least one element of `albedo` is outside of
    [0,1].
  """
  with tf.compat.v1.name_scope(name, "phong_brdf", [
      direction_incoming_light, direction_outgoing_light, surface_normal,
      shininess, albedo, brdf_normalization
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
    # Checks whether the incoming or outgoing light point behind the surface.
    dot_incoming_light_surface_normal = vector.dot(-direction_incoming_light,
                                                   surface_normal)
    dot_outgoing_light_surface_normal = vector.dot(direction_outgoing_light,
                                                   surface_normal)
    min_dot = tf.minimum(dot_incoming_light_surface_normal,
                         dot_outgoing_light_surface_normal)
    # BRDF evaluation.
    perfect_reflection_direction = vector.reflect(direction_incoming_light,
                                                  surface_normal)
    perfect_reflection_direction = tf.math.l2_normalize(
        perfect_reflection_direction, axis=-1)
    cos_alpha = vector.dot(
        perfect_reflection_direction, direction_outgoing_light, axis=-1)
    cos_alpha = tf.maximum(cos_alpha, tf.zeros_like(cos_alpha))
    phong_model = albedo * tf.pow(cos_alpha, shininess)
    if brdf_normalization:
      phong_model = phong_model * _brdf_normalization_factor(shininess)
    condition = tf.broadcast_to(
        tf.greater_equal(min_dot, 0.0), tf.shape(input=phong_model))
    return tf.where(condition, phong_model, tf.zeros_like(phong_model))


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
