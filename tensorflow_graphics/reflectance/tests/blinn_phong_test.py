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
"""Tests for Blinn-Phong reflectance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math as m
import sys

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.reflectance import blinn_phong
from tensorflow_graphics.util import test_case


class BlinnPhongTest(test_case.TestCase):

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_brdf_jacobian_random(self):
    """Tests the Jacobian of brdf."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    # Initialization.
    direction_incoming_light_init = np.random.uniform(
        -1.0, 1.0, size=tensor_shape + [3])
    direction_outgoing_light_init = np.random.uniform(
        -1.0, 1.0, size=tensor_shape + [3])
    surface_normal_init = np.random.uniform(-1.0, 1.0, size=tensor_shape + [3])
    shininess_init = np.random.uniform(size=tensor_shape + [1])
    albedo_init = np.random.random(tensor_shape + [3])
    # Conversion to tensors.
    direction_incoming_light = tf.convert_to_tensor(
        value=direction_incoming_light_init)
    direction_outgoing_light = tf.convert_to_tensor(
        value=direction_outgoing_light_init)
    surface_normal = tf.convert_to_tensor(value=surface_normal_init)
    shininess = tf.convert_to_tensor(value=shininess_init)
    albedo = tf.convert_to_tensor(value=albedo_init)
    y = blinn_phong.brdf(direction_incoming_light, direction_outgoing_light,
                         surface_normal, shininess, albedo)
    self.assert_jacobian_is_correct(direction_incoming_light,
                                    direction_incoming_light_init, y)
    self.assert_jacobian_is_correct(direction_outgoing_light,
                                    direction_outgoing_light_init, y)
    self.assert_jacobian_is_correct(surface_normal, surface_normal_init, y)
    self.assert_jacobian_is_correct(shininess, shininess_init, y)
    self.assert_jacobian_is_correct(albedo, albedo_init, y)

  def test_brdf_preset(self):
    """Tests that the brdf produces the expected results."""
    direction_incoming_light = np.random.uniform(-1.0, 1.0, size=(3,))
    direction_incoming_light /= np.linalg.norm(
        direction_incoming_light, axis=-1)
    direction_outgoing_light = np.random.uniform(-1.0, 1.0, size=(3,))
    direction_outgoing_light /= np.linalg.norm(
        direction_outgoing_light, axis=-1)
    surface_normal = np.random.uniform(-1.0, 1.0, size=(3,))
    surface_normal /= np.linalg.norm(surface_normal, axis=-1)
    shininess = np.array((0.0,))
    albedo = np.random.random(size=(3,))
    pred = blinn_phong.brdf(direction_incoming_light, direction_outgoing_light,
                            surface_normal, shininess, albedo)
    self.assertAllClose(albedo / m.pi, pred)

  def test_brdf_exceptions_raised(self):
    """Tests that the exceptions are raised correctly."""
    direction_incoming_light = np.random.uniform(-1.0, 1.0, size=(3,))
    direction_outgoing_light = np.random.uniform(-1.0, 1.0, size=(3,))
    surface_normal = np.random.uniform(-1.0, 1.0, size=(3,))
    shininess = np.random.uniform(0.0, 1.0, size=(1,))
    albedo = np.random.uniform(0.0, 1.0, (3,))
    with self.subTest(name="assert_on_direction_incoming_light_not_normalized"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            blinn_phong.brdf(direction_incoming_light, direction_outgoing_light,
                             surface_normal, shininess, albedo))
    direction_incoming_light /= np.linalg.norm(
        direction_incoming_light, axis=-1)
    with self.subTest(name="assert_on_direction_outgoing_light_not_normalized"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            blinn_phong.brdf(direction_incoming_light, direction_outgoing_light,
                             surface_normal, shininess, albedo))
    direction_outgoing_light /= np.linalg.norm(
        direction_outgoing_light, axis=-1)
    with self.subTest(name="assert_on_surface_normal_not_normalized"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            blinn_phong.brdf(direction_incoming_light, direction_outgoing_light,
                             surface_normal, shininess, albedo))
    surface_normal /= np.linalg.norm(surface_normal, axis=-1)
    with self.subTest(name="assert_on_albedo_not_normalized"):
      albedo = np.random.uniform(-10.0, -sys.float_info.epsilon, (3,))
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            blinn_phong.brdf(direction_incoming_light, direction_outgoing_light,
                             surface_normal, shininess, albedo))
      albedo = np.random.uniform(sys.float_info.epsilon, 10.0, (3,))
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            blinn_phong.brdf(direction_incoming_light, direction_outgoing_light,
                             surface_normal, shininess, albedo))

  @parameterized.parameters(
      ((3,), (3,), (3,), (1,), (3,)),
      ((None, 3), (None, 3), (None, 3), (None, 1), (None, 3)),
  )
  def test_brdf_shape_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(blinn_phong.brdf, shape)

  @parameterized.parameters(
      ("'direction_incoming_light' must have 3 dimensions.", (1,), (3,), (3,),
       (1,), (3,)),
      ("'direction_incoming_light' must have 3 dimensions.", (2,), (3,), (3,),
       (1,), (3,)),
      ("'direction_incoming_light' must have 3 dimensions.", (4,), (3,), (3,),
       (1,), (3,)),
      ("'direction_outgoing_light' must have 3 dimensions.", (3,), (1,), (3,),
       (1,), (3,)),
      ("'direction_outgoing_light' must have 3 dimensions.", (3,), (2,), (3,),
       (1,), (3,)),
      ("'direction_outgoing_light' must have 3 dimensions.", (3,), (4,), (3,),
       (1,), (3,)),
      ("'surface_normal' must have 3 dimensions.", (3,), (3,), (1,), (1,),
       (3,)),
      ("'surface_normal' must have 3 dimensions.", (3,), (3,), (2,), (1,),
       (3,)),
      ("'surface_normal' must have 3 dimensions.", (3,), (3,), (4,), (1,),
       (3,)),
      ("'shininess' must have 1 dimension.", (3,), (3,), (3,), (2,), (3,)),
      ("'shininess' must have 1 dimension.", (3,), (3,), (3,), (3,), (3,)),
      ("'albedo' must have 3 dimensions.", (3,), (3,), (3,), (1,), (4,)),
      ("'albedo' must have 3 dimensions.", (3,), (3,), (3,), (1,), (2,)),
      ("'albedo' must have 3 dimensions.", (3,), (3,), (3,), (1,), (1,)),
  )
  def test_brdf_shape_exception_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(blinn_phong.brdf, error_msg, shape)


if __name__ == "__main__":
  test_case.main()
