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
"""Tests for lambertian reflectance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math as m
import sys
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.reflectance import lambertian
from tensorflow_graphics.util import test_case


class LambertianTest(test_case.TestCase):

  def test_brdf_jacobian_random(self):
    """Tests the Jacobian of brdf."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    albedo_init = np.random.random(tensor_shape + [3])
    albedo = tf.convert_to_tensor(value=albedo_init)
    y = lambertian.brdf(albedo)
    self.assert_jacobian_is_correct(albedo, albedo_init, y)

  def test_brdf_random(self):
    """Tests that the brdf produces the expected results."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    albedo = np.random.uniform(0.0, 1.0, tensor_shape + [3])
    pred = lambertian.brdf(albedo)
    self.assertAllClose(albedo / m.pi, pred)

  def test_brdf_range_exception_raised(self):
    """Tests that the range exception is raised."""
    albedo = np.random.uniform(-10.0, -sys.float_info.epsilon, (3,))
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(lambertian.brdf(albedo))
    albedo = np.random.uniform(sys.float_info.epsilon, 10.0, (3,))
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(lambertian.brdf(albedo))

  @parameterized.parameters(
      ((3,)),
      ((None, 3),),
  )
  def test_brdf_shape_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(lambertian.brdf, shape)

  @parameterized.parameters(
      ("'albedo' must have 3 dimensions.", (4,)),
      ("'albedo' must have 3 dimensions.", (2,)),
      ("'albedo' must have 3 dimensions.", (1,)),
  )
  def test_brdf_shape_exception_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(lambertian.brdf, error_msg, shape)


if __name__ == "__main__":
  test_case.main()
