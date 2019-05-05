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
"""Tests for quadratic_radial_distortion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_graphics.rendering.camera import quadratic_radial_distortion
from tensorflow_graphics.util import test_case

RANDOM_TESTS_NUM_SAMPLES = 100


class QuadraticRadialDistortionTest(test_case.TestCase):

  def test_distortion_factor_random_positive_distortion_coefficient(self):
    """Tests that distortion_factor produces the expected outputs."""
    squared_radii = np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float32') * 2.0
    distortion_coefficient = np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float32') * 2.0
    distortion, mask = quadratic_radial_distortion.distortion_factor(
        squared_radii, distortion_coefficient)
    with self.subTest(name='distortion'):
      self.assertAllClose(1.0 + distortion_coefficient * squared_radii,
                          distortion)
    # No overflow when distortion_coefficient >= 0.0.
    with self.subTest(name='mask'):
      self.assertAllEqual([False] * RANDOM_TESTS_NUM_SAMPLES, mask)

  def test_distortion_factor_preset_zero_distortion_coefficient(self):
    """Tests distortion_factor at zero distortion coefficient."""
    squared_radii = np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float32') * 2.0
    distortion, mask = quadratic_radial_distortion.distortion_factor(
        squared_radii, 0.0)
    with self.subTest(name='distortion'):
      self.assertAllClose(tf.ones_like(squared_radii), distortion)
    # No overflow when distortion_coefficient = 0.0.
    self.assertAllEqual([False] * RANDOM_TESTS_NUM_SAMPLES, mask)

  def test_distortion_factor_random_negative_distortion_coefficient(self):
    """Tests that distortion_factor produces the expected outputs."""
    squared_radii = np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float32') * 2.0
    distortion_coefficient = -np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float32') * 0.2
    max_squared_radii = -1.0 / 3.0 / distortion_coefficient
    expected_overflow_mask = squared_radii > max_squared_radii
    valid_mask = np.logical_not(expected_overflow_mask)

    distortion, mask = quadratic_radial_distortion.distortion_factor(
        squared_radii, distortion_coefficient)

    # We assert correctness of the mask, and of all the pixels that are not in
    # overflow.
    actual_distortion_when_valid = self.evaluate(distortion)[valid_mask]
    expected_distortion_when_valid = (
        1.0 + distortion_coefficient * squared_radii[valid_mask])

    with self.subTest(name='distortion'):
      self.assertAllClose(expected_distortion_when_valid,
                          actual_distortion_when_valid)
    with self.subTest(name='mask'):
      self.assertAllEqual(expected_overflow_mask, mask)

  def test_distortion_factor_preset_zero_radius(self):
    """Tests distortion_factor at the corner case of zero radius."""
    distortion_coefficient = np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float32') - 0.5
    squared_radii = tf.zeros_like(distortion_coefficient)
    distortion, mask = quadratic_radial_distortion.distortion_factor(
        squared_radii, distortion_coefficient)

    with self.subTest(name='distortion'):
      self.assertAllClose([1.0] * RANDOM_TESTS_NUM_SAMPLES, distortion)
    with self.subTest(name='mask'):
      self.assertAllEqual([False] * RANDOM_TESTS_NUM_SAMPLES, mask)

  @parameterized.parameters(quadratic_radial_distortion.distortion_factor,
                            quadratic_radial_distortion.undistortion_factor)
  def test_both_negative_radius_exception_raised(self, distortion_function):
    """Tests that an exception is raised when the squared radius is negative."""
    distortion_coefficient = np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float32') - 0.5
    squared_radii = [-0.5] * RANDOM_TESTS_NUM_SAMPLES
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(distortion_function(squared_radii, distortion_coefficient))

  @parameterized.parameters((2, 2e-3), (3, 1e-8))
  def test_undistortion_factor_random_positive_distortion_coefficient(
      self, num_iterations, tolerance):
    """Tests that undistortion_factor produces the expected outputs."""
    distorted_squared_radii = np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float32') * 2.0
    distortion_coefficient = np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float32') * 0.2
    undistortion, mask = quadratic_radial_distortion.undistortion_factor(
        distorted_squared_radii, distortion_coefficient, num_iterations)
    undistorted_squared_radii = tf.square(
        undistortion) * distorted_squared_radii
    # We distort again the undistorted radii and compare to the original
    # distorted_squared_radii.
    redistorted_squared_radii = tf.square(
        1.0 + distortion_coefficient *
        undistorted_squared_radii) * undistorted_squared_radii
    with self.subTest(name='distortion'):
      self.assertAllClose(
          distorted_squared_radii, redistorted_squared_radii, atol=tolerance)

    # Positive distortion_coefficient-s never overflow.
    with self.subTest(name='mask'):
      self.assertAllEqual([False] * RANDOM_TESTS_NUM_SAMPLES, mask)

  @parameterized.parameters((2, 1e-2), (3, 6e-4), (4, 2e-5))
  def test_undistortion_factor_random_negative_distortion_coefficient(
      self, num_iterations, tolerance):
    """Tests that undistortion_factor produces the expected outputs."""
    distorted_squared_radii = np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float32') * 2.0
    distortion_coefficient = -np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float32') * 0.2
    undistortion, mask = quadratic_radial_distortion.undistortion_factor(
        distorted_squared_radii, distortion_coefficient, num_iterations)
    undistorted_squared_radii = tf.square(
        undistortion) * distorted_squared_radii
    # See explanation in the implementation comments for this formula
    expected_overflow_mask = (
        distorted_squared_radii * distortion_coefficient + 4.0 / 27.0 < 0)
    redistorted_squared_radii = tf.square(
        1.0 + distortion_coefficient *
        undistorted_squared_radii) * undistorted_squared_radii
    valid_mask = np.logical_not(expected_overflow_mask)
    redistorted_squared_radii_when_valid = self.evaluate(
        redistorted_squared_radii)[valid_mask]
    distorted_squared_radii_when_valid = distorted_squared_radii[valid_mask]

    with self.subTest(name='distortion'):
      self.assertAllClose(
          distorted_squared_radii_when_valid,
          redistorted_squared_radii_when_valid,
          atol=tolerance)
    # We assert correctness of the mask, and of all the pixels that are not in
    # overflow, distorting again the undistorted radii and comparing to the
    # original distorted_squared_radii
    with self.subTest(name='mask'):
      self.assertAllEqual(expected_overflow_mask, mask)

  def test_undistortion_factor_zero_distortion_coefficient(self):
    """Tests undistortion_factor at zero distortion coefficient."""
    squared_radii = np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float32') * 2.0
    undistortion, mask = quadratic_radial_distortion.undistortion_factor(
        squared_radii, 0.0)

    with self.subTest(name='distortion'):
      self.assertAllClose(tf.ones_like(squared_radii), undistortion)
    # No overflow when distortion_coefficient = 0.0
    with self.subTest(name='mask'):
      self.assertAllEqual([False] * RANDOM_TESTS_NUM_SAMPLES, mask)

  @parameterized.parameters(quadratic_radial_distortion.distortion_factor,
                            quadratic_radial_distortion.undistortion_factor)
  def test_broadcasting(self, distortion_function):
    """Tests shape broadcasting."""
    shape = [2, 5, 7, 11]
    for rank in range(len(shape) + 1):
      squared_radii = np.random.rand(*shape).astype('float32')
      coeffs_shape = shape[:rank]
      coeffs = np.random.rand(*coeffs_shape) * 0.2 - 0.1
      coeffs = coeffs.astype('float32') if rank > 0 else float(coeffs)

      coeffs = tf.reshape(coeffs,
                          coeffs_shape + [1] * (len(shape) - len(coeffs_shape)))
      broadcast_coeffs = tf.broadcast_to(coeffs, shape)
      result_with_broadcast = distortion_function(squared_radii,
                                                  broadcast_coeffs)
      result = distortion_function(squared_radii, coeffs)
      with self.subTest(name='distortion'):
        self.assertAllClose(result_with_broadcast[0], result[0])
      self.assertAllEqual(result_with_broadcast[1], result[1])

  @parameterized.parameters(quadratic_radial_distortion.distortion_factor,
                            quadratic_radial_distortion.undistortion_factor)
  def test_both_radial_jacobian(self, distortion_function):
    """Test the Jacobians with respect to squared radii."""
    squared_radii = np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float64') * 0.5
    distortion_coefficients = np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float64') * 0.6 - 0.3
    squared_radii_tensor = tf.convert_to_tensor(
        value=squared_radii, dtype=tf.float64)
    distortion, _ = distortion_function(squared_radii_tensor,
                                        distortion_coefficients)
    self.assert_jacobian_is_correct(squared_radii_tensor, squared_radii,
                                    distortion)

  @parameterized.parameters(quadratic_radial_distortion.distortion_factor,
                            quadratic_radial_distortion.undistortion_factor)
  def test_both_distortion_coefficient_jacobian(self, distortion_function):
    """Test the Jacobians with respect to distortion coefficients."""
    distortion_coefficients = np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float64') * 0.6 - 0.3
    squared_radii = np.random.rand(RANDOM_TESTS_NUM_SAMPLES).astype(
        'float64') * 0.5
    distortion_coefficients_tensor = tf.convert_to_tensor(
        value=distortion_coefficients)
    distortion, _ = distortion_function(squared_radii,
                                        distortion_coefficients_tensor)
    self.assert_jacobian_is_correct(distortion_coefficients_tensor,
                                    distortion_coefficients, distortion)


if __name__ == '__main__':
  test_case.main()
