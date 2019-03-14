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
"""Tests for shape utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.util import shape
from tensorflow_graphics.util import test_case


class ShapeTest(test_case.TestCase):

  @parameterized.parameters(
      (None, None, False),
      ((2, 3, 5, 7), (2, 3, 5, 7), True),
      ((1, 3, 5, 7), (2, 3, 5, 7), True),
      ((2, 3, 5, 7), (1, 3, 5, 7), True),
      ((None, 3, 5, 7), (None, 3, 5, 7), True),
      ((2, None, 5, 7), (2, 3, None, 7), True),
      ((1, 3, None, 7), (2, 3, 5, None), True),
      ((None, 3, 5, 7), (None, 3, 5, 7), True),
      ((None, 3, 5, 7), (None, 3, 5, 1), True),
      ((None, 3, 5, 7), (None, 3, 1, 1), True),
      ((None, 3, 5, 7), (None, 1, 1, 1), True),
      ((None, 3, 5, 7), (2, 3, 5, 7), True),
      ((None, 3, 5, 7), (1, 3, 5, 7), True),
      ((None, 3, 5, 7), (None, 5, 7), True),
      ((None, 3, 5, 7), (None, 7), True),
      ((None, 3, 5, 7), (None, 1, 1), True),
      ((None, 3, 5, 7), (None, 1), True),
      ((None, 3, 5, 7), (None,), True),
      ((2, 3, 5, 7), (3, 3, 5, 7), False),
      ((3, 3, 5, 7), (2, 3, 5, 7), False),
      ((None, 3, 5, 7), (None, 3, 5), False),
      ((None, 3, 5, 7), (None, 5), False),
      ((None, 3, 5, 7), (None, 3, 5, 7, 1), False),
      ((None, 3, 5, 7), (None, 2, 5, 7), False),
      ((None, 3, 5, 7), (None, 3, 4, 7), False),
      ((None, 3, 5, 7), (None, 3, 5, 6), False),
  )
  def test_is_broadcast_compatible(self, shape_x, shape_y, broadcastable):
    """Checks if the is_broadcast_compatible function works as expected."""
    if tf.executing_eagerly():
      if (shape_x is None or shape_y is None or None in shape_x or
          None in shape_y):
        return
      shape_x = tf.compat.v1.placeholder_with_default(
          tf.zeros(shape_x, dtype=tf.float32), shape=shape_x).shape
      shape_y = tf.compat.v1.placeholder_with_default(
          tf.zeros(shape_y, dtype=tf.float32), shape=shape_y).shape
    else:
      shape_x = tf.compat.v1.placeholder(shape=shape_x, dtype=tf.float32).shape
      shape_y = tf.compat.v1.placeholder(shape=shape_y, dtype=tf.float32).shape
    self.assertEqual(
        shape.is_broadcast_compatible(shape_x, shape_y), broadcastable)


if __name__ == "__main__":
  test_case.main()
