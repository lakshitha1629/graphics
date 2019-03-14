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
"""Tests for asserts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import test_case


def _pick_random_vector():
  """Creates a random vector with a random shape."""
  tensor_size = np.random.randint(3)
  tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
  return np.random.normal(size=tensor_shape + [4])


class AssertsTest(test_case.TestCase):

  @parameterized.parameters((tf.float16,), (tf.float32,), (tf.float64,))
  def test_assert_normalized(self, dtype):
    """Checks that assert_normalized function works as expected."""
    vector = _pick_random_vector()
    vector = tf.convert_to_tensor(value=vector, dtype=dtype)
    norm_vector = vector / tf.norm(tensor=vector, axis=-1, keepdims=True)
    norm_vector = asserts.assert_normalized(norm_vector)
    self.evaluate(norm_vector)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      vector = asserts.assert_normalized(vector)
      self.evaluate(vector)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_assert_normalized_passthrough(self):
    """Checks that the assert is a passthrough when the flag is False."""
    vector_input = _pick_random_vector()
    vector_output = asserts.assert_normalized(vector_input)
    self.assertTrue(vector_input is vector_output)

  def test_assert_nonzero_norm(self):
    """Checks whether assert_nonzero_norm works for almost zero vectors."""
    vector = _pick_random_vector()
    zero_vector = tf.zeros_like(vector)
    vector = asserts.assert_nonzero_norm(vector)
    self.evaluate(vector)
    self.evaluate(asserts.assert_nonzero_norm(tf.constant([4e-4], tf.float16)))
    self.evaluate(asserts.assert_nonzero_norm(tf.constant([4e-19], tf.float32)))
    self.evaluate(
        asserts.assert_nonzero_norm(tf.constant([4e-154], tf.float64)))
    with self.assertRaises(tf.errors.InvalidArgumentError):
      zero_vector = asserts.assert_nonzero_norm(zero_vector)
      self.evaluate(zero_vector)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          asserts.assert_nonzero_norm(tf.constant([1e-4], tf.float16)))
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          asserts.assert_nonzero_norm(tf.constant([1e-38], tf.float32)))
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          asserts.assert_nonzero_norm(tf.constant([1e-308], tf.float64)))

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_assert_nonzero_norm_passthrough(self):
    """Checks that the assert is a passthrough when the flag is False."""
    vector_input = _pick_random_vector()
    vector_output = asserts.assert_nonzero_norm(vector_input)
    self.assertTrue(vector_input is vector_output)

  @parameterized.parameters((tf.float16,), (tf.float32,), (tf.float64,))
  def test_assert_all_above(self, dtype):
    """Checks whether assert_all_above works as intended."""
    vector = tf.convert_to_tensor(value=_pick_random_vector(), dtype=dtype)
    vector = vector * vector
    vector /= -tf.reduce_max(input_tensor=vector, axis=-1, keepdims=True)
    eps = asserts.select_eps_for_addition(dtype)
    inside_vector = vector + eps
    outside_vector = vector - eps
    ones_vector = -tf.ones_like(vector)
    inside_vector_open = asserts.assert_all_above(
        inside_vector, -1.0, open_bound=True)
    inside_vector_closed = asserts.assert_all_above(
        inside_vector, -1.0, open_bound=False)
    ones_vector_closed = asserts.assert_all_above(
        ones_vector, -1.0, open_bound=False)
    self.evaluate(inside_vector_open)
    self.evaluate(inside_vector_closed)
    self.evaluate(ones_vector_closed)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      outside_vector_open = asserts.assert_all_above(
          outside_vector, -1.0, open_bound=True)
      self.evaluate(outside_vector_open)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      outside_vector_closed = asserts.assert_all_above(
          outside_vector, -1.0, open_bound=False)
      self.evaluate(outside_vector_closed)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      ones_vector_open = asserts.assert_all_above(
          ones_vector, -1.0, open_bound=True)
      self.evaluate(ones_vector_open)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_assert_all_above_passthrough(self):
    """Checks that the assert is a passthrough when the flag is False."""
    vector_input = _pick_random_vector()
    vector_output = asserts.assert_all_above(vector_input, 1.0)
    self.assertTrue(vector_input is vector_output)

  @parameterized.parameters((tf.float16,), (tf.float32,), (tf.float64,))
  def test_assert_all_below(self, dtype):
    """Checks whether assert_all_below works as intended."""
    vector = tf.convert_to_tensor(value=_pick_random_vector(), dtype=dtype)
    vector = vector * vector
    vector /= tf.reduce_max(input_tensor=vector, axis=-1, keepdims=True)
    eps = asserts.select_eps_for_addition(dtype)
    inside_vector = vector - eps
    outside_vector = vector + eps
    ones_vector = tf.ones_like(vector)
    inside_vector_open = asserts.assert_all_below(
        inside_vector, 1.0, open_bound=True)
    inside_vector_closed = asserts.assert_all_below(
        inside_vector, 1.0, open_bound=False)
    ones_vector_closed = asserts.assert_all_below(
        ones_vector, 1.0, open_bound=False)
    self.evaluate(inside_vector_open)
    self.evaluate(inside_vector_closed)
    self.evaluate(ones_vector_closed)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      outside_vector_open = asserts.assert_all_below(
          outside_vector, 1.0, open_bound=True)
      self.evaluate(outside_vector_open)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      outside_vector_closed = asserts.assert_all_below(
          outside_vector, 1.0, open_bound=False)
      self.evaluate(outside_vector_closed)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      ones_vector_open = asserts.assert_all_below(
          ones_vector, 1.0, open_bound=True)
      self.evaluate(ones_vector_open)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_assert_all_below_passthrough(self):
    """Checks that the assert is a passthrough when the flag is False."""
    vector_input = _pick_random_vector()
    vector_output = asserts.assert_all_below(vector_input, 0.0)
    self.assertTrue(vector_input is vector_output)

  @parameterized.parameters((tf.float16,), (tf.float32,), (tf.float64,))
  def test_assert_all_in_range(self, dtype):
    """Checks whether assert_all_in_range works as intended."""
    vector = tf.convert_to_tensor(value=_pick_random_vector(), dtype=dtype)
    vector = vector * vector
    vector /= tf.reduce_max(input_tensor=vector, axis=-1, keepdims=True)
    eps = asserts.select_eps_for_addition(dtype)
    inside_vector = vector - eps
    outside_vector = vector + eps
    ones_vector = tf.ones_like(vector)
    inside_vector_open = asserts.assert_all_in_range(
        inside_vector, -1.0, 1.0, open_bounds=True)
    inside_vector_closed = asserts.assert_all_in_range(
        inside_vector, -1.0, 1.0, open_bounds=False)
    ones_vector_closed = asserts.assert_all_in_range(
        ones_vector, -1.0, 1.0, open_bounds=False)
    self.evaluate(inside_vector_open)
    self.evaluate(inside_vector_closed)
    self.evaluate(ones_vector_closed)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      outside_vector_open = asserts.assert_all_in_range(
          outside_vector, -1.0, 1.0, open_bounds=True)
      self.evaluate(outside_vector_open)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      outside_vector_closed = asserts.assert_all_in_range(
          outside_vector, -1.0, 1.0, open_bounds=False)
      self.evaluate(outside_vector_closed)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      ones_vector_open = asserts.assert_all_in_range(
          ones_vector, -1.0, 1.0, open_bounds=True)
      self.evaluate(ones_vector_open)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_assert_all_in_range_passthrough(self):
    """Checks that the assert is a passthrough when the flag is False."""
    vector_input = _pick_random_vector()
    vector_output = asserts.assert_all_in_range(vector_input, -1.0, 1.0)
    self.assertTrue(vector_input is vector_output)

  def test_select_eps_for_division_arguments(self):
    """Checks if select_eps_for_division raises ValueError for non-floats."""
    some_int = tf.constant([1], tf.int32)
    with self.assertRaises(ValueError):
      asserts.select_eps_for_division(some_int.dtype)

  def test_select_eps_for_addition_arguments(self):
    """Checks if select_eps_for_addition raises ValueError for non-floats."""
    some_int = tf.constant([1], tf.int32)
    with self.assertRaises(ValueError):
      asserts.select_eps_for_addition(some_int.dtype)

  @parameterized.parameters((tf.float16,), (tf.float32,), (tf.float64,))
  def test_select_eps_for_division(self, dtype):
    """Checks whether select_eps_for_division causes Inf values."""
    a = tf.constant(1.0, dtype=dtype)
    eps = asserts.select_eps_for_division(dtype)
    eps = tf.convert_to_tensor(value=eps, dtype=dtype)
    self.evaluate(tf.debugging.check_numerics(a / eps, "Inf detected."))

  @parameterized.parameters((tf.float16,), (tf.float32,), (tf.float64,))
  def test_select_eps_for_addition(self, dtype):
    """Checks whether select_eps_for_addition returns large enough eps."""
    a = tf.constant(1.0, dtype=dtype)
    eps = asserts.select_eps_for_addition(dtype)
    eps = tf.convert_to_tensor(value=eps, dtype=dtype)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(tf.compat.v1.assert_equal(a, eps))


if __name__ == "__main__":
  test_case.main()
