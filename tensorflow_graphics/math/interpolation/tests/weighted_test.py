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
"""Tests for google3.third_party.py.tensorflow_graphics.interpolation.weighted."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.math.interpolation import weighted
from tensorflow_graphics.util import test_case


class WeightedTest(test_case.TestCase):

  @parameterized.parameters(
      (3, 4, 2, 3),
      (5, 4, 5, 3),
      (5, 6, 5, 5),
      (2, 6, 5, 1),
  )
  def test_interpolate_not_raised(self, dim_points, num_points, num_outputs,
                                  num_pts_to_interpolate):
    """Tests whether exceptions are not raise for compatible shapes."""
    points_shape = (num_points, dim_points)
    weights_shape = (num_outputs, num_pts_to_interpolate)
    indices = tf.constant([
        np.random.permutation(num_points)[:num_pts_to_interpolate].tolist()
        for _ in range(num_outputs)
    ])
    indices = tf.expand_dims(indices, axis=-1)
    points = tf.constant(np.random.uniform(size=points_shape))
    weights = tf.constant(np.random.uniform(size=weights_shape))
    result = weighted.interpolate(
        points=points, weights=weights, indices=indices, normalize=True)

    self.assertSequenceEqual((num_outputs, dim_points),
                             result.get_shape().as_list())

  @parameterized.parameters(
      (((-1.0, 1.0), (1.0, 1.0), (3.0, 1.0), (-1.0, -1.0), (1.0, -1.0),
        (3.0, -1.0)), ((0.25, 0.25, 0.25, 0.25), (0.5, 0.5, 0.0, 0.0)),
       (((0,), (1,), (3,), (4,)), ((1,), (2,), (4,),
                                   (5,))), False, ((0.0, 0.0), (2.0, 1.0))))
  def test_interpolate_preset(self, points, weights, indices, normalize, out):
    """Tests whether interpolation results are correct."""
    weights = tf.convert_to_tensor(value=weights)
    result_unnormalized = weighted.interpolate(
        points=points, weights=weights, indices=indices, normalize=False)
    result_normalized = weighted.interpolate(
        points=points, weights=2.0 * weights, indices=indices, normalize=True)
    estimated_unnormalized = self.evaluate(result_unnormalized)
    estimated_normalized = self.evaluate(result_normalized)
    self.assertAllClose(estimated_unnormalized, out)
    self.assertAllClose(estimated_normalized, out)

  @parameterized.parameters(
      (3, 4, 2, 3),
      (5, 4, 5, 3),
      (5, 6, 5, 5),
      (2, 6, 5, 1),
  )
  def test_interpolate_negative_weights_raised(
      self, dim_points, num_points, num_outputs, num_pts_to_interpolate):
    """Tests whether exception is raised when weights are negative."""
    points_shape = (num_points, dim_points)
    weights_shape = (num_outputs, num_pts_to_interpolate)
    indices = tf.constant([
        np.random.permutation(num_points)[:num_pts_to_interpolate].tolist()
        for _ in range(num_outputs)
    ])
    indices = tf.expand_dims(indices, axis=-1)
    points = tf.constant(np.random.uniform(size=points_shape))
    weights = -1.0 * tf.constant(np.random.uniform(size=weights_shape))
    with self.assertRaises(tf.errors.InvalidArgumentError):
      result = weighted.interpolate(
          points=points, weights=weights, indices=indices, normalize=True)
      self.evaluate(result)

  @parameterized.parameters(
      (((-1.0, 1.0), (1.0, 1.0), (3.0, 1.0), (-1.0, -1.0), (1.0, -1.0),
        (3.0, -1.0)), ((1.0, -1.0, 1.0, -1.0), (0.0, 0.0, 0.0, 0.0)),
       (((0,), (1,), (3,), (4,)), ((1,), (2,), (4,), (5,))), ((0.0, 0.0),
                                                              (0.0, 0.0))))
  def test_interpolate_unnormalizable_raised_(self, points, weights, indices,
                                              out):
    """Tests whether exception is raised when weights are unnormalizable."""
    with self.assertRaises(tf.errors.InvalidArgumentError):
      result = weighted.interpolate(
          points=points,
          weights=weights,
          indices=indices,
          normalize=True,
          allow_negative_weights=True)
      self.evaluate(result)

  @parameterized.parameters(
      (3, 4, 2, 3),
      (5, 4, 5, 3),
      (5, 6, 5, 5),
      (2, 6, 5, 1),
  )
  def test_interpolate_jacobian_random(self, dim_points, num_points,
                                       num_outputs, num_pts_to_interpolate):
    """Tests whether jacobian is correct."""
    points_shape = (num_points, dim_points)
    weights_shape = (num_outputs, num_pts_to_interpolate)
    indices_np = np.array([
        np.random.permutation(num_points)[:num_pts_to_interpolate].tolist()
        for _ in range(num_outputs)
    ])
    indices_np = np.expand_dims(indices_np, axis=-1)
    indices = tf.convert_to_tensor(value=indices_np)
    points_np = np.random.uniform(size=points_shape)
    points = tf.convert_to_tensor(value=points_np)
    weights_np = np.random.uniform(size=weights_shape)
    weights = tf.convert_to_tensor(value=weights_np)

    y = weighted.interpolate(
        points=points, weights=weights, indices=indices, normalize=True)

    with self.subTest(name="points"):
      self.assert_jacobian_is_correct(points, points_np, y)
    with self.subTest(name="weights"):
      self.assert_jacobian_is_correct(weights, weights_np, y)


if __name__ == "__main__":
  test_case.main()
