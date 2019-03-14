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
r"""Tests for ray.

blaze test \
third_party/py/tensorflow_graphics/geometry/ray_test \
--test_output=all
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry import ray
from tensorflow_graphics.util import test_case

FLAGS = flags.FLAGS


class RayTest(test_case.TestCase):

  def setUp(self):
    """Sets the seed for tensorflow and numpy."""
    super(test_case.TestCase, self).setUp()

  def _generate_random_example(self):
    num_cameras = 4
    num_keypoints = 3
    batch_size = 2
    self.points_values = np.random.random_sample((batch_size, num_keypoints, 3))
    points_expanded_values = np.expand_dims(self.points_values, axis=-2)
    startpoints_values = np.random.random_sample(
        (batch_size, num_keypoints, num_cameras, 3))

    difference = points_expanded_values - startpoints_values
    difference_norm = np.sqrt((difference * difference).sum(axis=-1))
    direction = difference / np.expand_dims(difference_norm, axis=-1)

    self.startpoints_values = points_expanded_values - 0.5 * direction
    self.endpoints_values = points_expanded_values + 0.5 * direction
    self.weights_values = np.ones((batch_size, num_keypoints, num_cameras))

    self.points = tf.convert_to_tensor(value=self.points_values)
    self.startpoints = tf.convert_to_tensor(value=self.startpoints_values)
    self.endpoints = tf.convert_to_tensor(value=self.endpoints_values)
    self.weights = tf.convert_to_tensor(value=self.weights_values)

  @parameterized.parameters(
      ("'startpoints' and 'endpoints' must have the same shape.", (4, 3),
       (5, 3), (4,)),
      ("'startpoints' and 'endpoints' must have their last dimension set to 3.",
       (4, 2), (4, 2), (4,)),
      ("'startpoints' and 'endpoints' must have at least two dimensions.", (3,),
       (3,), (None,)),
      ("'startpoints' and 'endpoints' must have their penultimate dimension "
       ">= 2.", (1, 3), (1, 3), (1,)),
      ("'weights' should have the same shape as 'startpoints' and 'endpoints', "
       "except that the last dimension should not be present in 'weights'.",
       (2, 4, 3), (2, 4, 3), (2, 5)),
  )
  def test_triangulate_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(ray.triangulate, error_msg, shapes)

  @parameterized.parameters(
      ((4, 3,), (4, 3,), (4,)),
      ((5, 4, 3,), (5, 4, 3,), (5, 4,)),
      ((6, 5, 4, 3,), (6, 5, 4, 3,), (6, 5, 4,)),
  )
  def test_triangulate_exception_is_not_raised(self, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_not_raised(ray.triangulate, shapes)

  def test_triangulate_jacobian_is_correct(self):
    """Tests that Jacobian is correct."""
    self._generate_random_example()

    points = ray.triangulate(self.startpoints, self.endpoints, self.weights)

    self.assert_jacobian_is_correct(
        self.startpoints, self.startpoints_values, points)
    self.assert_jacobian_is_correct(
        self.endpoints, self.endpoints_values, points)
    self.assert_jacobian_is_correct(
        self.weights, self.weights_values, points)

  def test_triangulate_jacobian_is_finite(self):
    """Tests that Jacobian is finite."""
    self._generate_random_example()

    points = ray.triangulate(self.startpoints, self.endpoints, self.weights)

    self.assert_jacobian_is_finite(
        self.startpoints, self.startpoints_values, points)
    self.assert_jacobian_is_finite(
        self.endpoints, self.endpoints_values, points)
    self.assert_jacobian_is_finite(
        self.weights, self.weights_values, points)

  def test_triangulate_random(self):
    """Tests that original points are recovered by triangualtion."""
    self._generate_random_example()

    test_inputs = (self.startpoints, self.endpoints, self.weights)
    test_outputs = (self.points_values,)
    self.assert_output_is_correct(
        ray.triangulate, test_inputs, test_outputs, rtol=1e-05, atol=1e-08,
        tile=False)


if __name__ == "__main__":
  test_case.main()
