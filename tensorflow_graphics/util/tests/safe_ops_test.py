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
"""Tests for safe_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import safe_ops
from tensorflow_graphics.util import test_case


def _pick_random_vector():
  """Creates a random vector with random shape."""
  tensor_size = np.random.randint(3)
  tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
  return np.random.normal(size=tensor_shape + [4])


class SafeOpsTest(test_case.TestCase):

  @parameterized.parameters((tf.float16), (tf.float32), (tf.float64))
  def test_safe_unsigned_div(self, dtype):
    """Checks if unsigned division can cause Inf values."""
    vector = tf.convert_to_tensor(value=_pick_random_vector(), dtype=dtype)
    zero_vector = tf.zeros_like(vector)
    div = safe_ops.safe_unsigned_div(
        tf.norm(tensor=vector), tf.norm(tensor=zero_vector))
    self.evaluate(div)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      unsafe_div = safe_ops.safe_unsigned_div(
          tf.norm(tensor=vector), tf.norm(tensor=zero_vector), eps=0.0)
      self.evaluate(unsafe_div)

  @parameterized.parameters((tf.float16), (tf.float32), (tf.float64))
  def test_safe_signed_div(self, dtype):
    """Checks if safe signed divisions can cause Inf values."""
    vector = tf.convert_to_tensor(value=_pick_random_vector(), dtype=dtype)
    zero_vector = tf.zeros_like(vector)
    div = safe_ops.safe_signed_div(tf.norm(tensor=vector), tf.sin(zero_vector))
    self.evaluate(div)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      unsafe_div = safe_ops.safe_unsigned_div(
          tf.norm(tensor=vector), tf.sin(zero_vector), eps=0.0)
      self.evaluate(unsafe_div)

  @parameterized.parameters((tf.float32), (tf.float64))
  def test_safe_shrink(self, dtype):
    """Checks whether safe shrinking makes tensor safe for tf.acos(x)."""
    tensor = tf.convert_to_tensor(value=_pick_random_vector(), dtype=dtype)
    tensor = tensor * tensor
    norm_tensor = tensor / tf.reduce_max(
        input_tensor=tensor, axis=-1, keepdims=True)
    eps = asserts.select_eps_for_addition(dtype)
    norm_tensor += eps
    with self.assertRaises(ValueError):
      safe_ops.safe_shrink(norm_tensor, eps=0.0)
    safe_tensor = safe_ops.safe_shrink(norm_tensor, -1.0, 1.0)
    self.evaluate(tf.acos(safe_tensor))
    with self.assertRaises(tf.errors.InvalidArgumentError):
      unsafe_tensor = safe_ops.safe_shrink(norm_tensor, -1.0, 1.0, eps=0.0)
      self.evaluate(tf.acos(unsafe_tensor))

  def test_safe_sinpx_div_sinx(self):
    """Tests for edge cases and continuity for sin(px)/sin(x)."""
    # Continuity tests for various angles.
    angle_step = np.pi / 16.0
    theta = tf.range(-2.0 * np.pi, 2.0 * np.pi + angle_step / 2.0, angle_step)
    p = np.random.uniform(size=[1])
    division = safe_ops.safe_sinpx_div_sinx(theta, p)
    division_l = safe_ops.safe_sinpx_div_sinx(theta + 1e-10, p)
    division_r = safe_ops.safe_sinpx_div_sinx(theta - 1e-10, p)
    self.assertAllClose(division, division_l, rtol=1e-9)
    self.assertAllClose(division, division_r, rtol=1e-9)

    # Tests for theta = 0, which causes zero / zero.
    theta = 0.0
    p = tf.range(0.0, 1.0, 0.001)
    division = safe_ops.safe_sinpx_div_sinx(theta, p)
    division_l = safe_ops.safe_sinpx_div_sinx(theta + 1e-10, p)
    division_r = safe_ops.safe_sinpx_div_sinx(theta - 1e-10, p)
    self.assertAllClose(division, division_l, atol=1e-9)
    self.assertAllClose(division, division_r, atol=1e-9)
    # According to l'Hopital rule, limit should be p
    self.assertAllClose(division, p, atol=1e-9)

    # Tests for theta = pi, which causes division by zero.
    theta = np.pi
    p = tf.range(0.0, 1.001, 0.001)
    division = safe_ops.safe_sinpx_div_sinx(theta, p)
    division_l = safe_ops.safe_sinpx_div_sinx(theta + 1e-10, p)
    division_r = safe_ops.safe_sinpx_div_sinx(theta - 1e-10, p)
    self.assertAllClose(division, division_l, atol=1e-9)
    self.assertAllClose(division, division_r, atol=1e-9)

  def test_safe_cospx_div_cosx(self):
    """Tests for edge cases and continuity for cos(px)/cos(x)."""
    # Continuity tests for various angles.
    angle_step = np.pi / 16.0
    theta = tf.range(-2.0 * np.pi, 2.0 * np.pi + angle_step / 2.0, angle_step)
    p = np.random.uniform(size=[1])
    division = safe_ops.safe_cospx_div_cosx(theta, p)
    division_l = safe_ops.safe_cospx_div_cosx(theta + 1e-10, p)
    division_r = safe_ops.safe_cospx_div_cosx(theta - 1e-10, p)
    self.assertAllClose(division, division_l, rtol=1e-9)
    self.assertAllClose(division, division_r, rtol=1e-9)

    # The case with 0 / 0 - should return 1.0
    theta = np.pi / 2.0
    p = tf.constant(1.0)
    division = safe_ops.safe_cospx_div_cosx(theta, p)
    division_l = safe_ops.safe_cospx_div_cosx(theta + 1e-10, p)
    division_r = safe_ops.safe_cospx_div_cosx(theta - 1e-10, p)
    self.assertAllClose(division, division_l, atol=1e-9)
    self.assertAllClose(division, division_r, atol=1e-9)
    self.assertAllClose(division, 1.0, atol=1e-9)

    # Tests for theta = 3/2 pi, which causes division by zero.
    theta = np.pi * 3.0 / 2.0
    p = tf.range(0.0, 1.001, 0.001)
    division = safe_ops.safe_cospx_div_cosx(theta, p)
    division_l = safe_ops.safe_cospx_div_cosx(theta + 1e-10, p)
    division_r = safe_ops.safe_cospx_div_cosx(theta - 1e-10, p)
    self.assertAllClose(division, division_l, atol=1e-9)
    self.assertAllClose(division, division_r, atol=1e-9)


if __name__ == "__main__":
  test_case.main()
