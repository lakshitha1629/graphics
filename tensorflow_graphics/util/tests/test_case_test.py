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

import tensorflow as tf

from tensorflow_graphics.util import test_case


class TestCaseTest(test_case.TestCase):

  def _dummy_tf_lite_compatible_function(self, data):
    data = tf.convert_to_tensor(value=data)
    return 2.0 * data

  def test_assert_tf_lite_convertible(self):
    """Tests that assert_tf_lite_convertible succeeds with a simple function."""
    if tf.executing_eagerly():
      return
    tc = test_case.TestCase()
    tc.assert_tf_lite_convertible(
        func=self._dummy_tf_lite_compatible_function, shapes=((1,),))
    tc.assert_tf_lite_convertible(
        func=self._dummy_tf_lite_compatible_function,
        shapes=[[1]],
        test_inputs=((1.0,),))


if __name__ == "__main__":
  test_case.main()
