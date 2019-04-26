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
"""Tests for utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow_graphics.geometry.representation.mesh import utils
from tensorflow_graphics.util import test_case


class UtilsTest(test_case.TestCase):

  @parameterized.parameters(
      (np.array(((0, 1, 2),)), [[0, 1], [0, 2], [1, 2]]),
      (np.array(
          ((0, 1, 2), (0, 1, 3))), [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]),
  )
  def test_extract_unique_edges_from_triangular_mesh_preset(
      self, test_inputs, test_outputs):
    """Tests that the output contain the expected edges."""
    self.assertEqual(
        sorted(
            utils.extract_unique_edges_from_triangular_mesh(
                test_inputs).tolist()), test_outputs)

  @parameterized.parameters(
      (1, "'faces' must be a numpy.ndarray."),
      (np.array((1,)), "must have a rank equal to 2"),
      (np.array((((1,),),)), "must have a rank equal to 2"),
      (np.array(((1,),)), "must have exactly 3 dimensions in the last axis"),
      (np.array(((1, 1),)), "must have exactly 3 dimensions in the last axis"),
      (np.array(
          ((1, 1, 1, 1),)), "must have exactly 3 dimensions in the last axis"),
  )
  def test_extract_unique_edges_from_triangular_mesh_raised(
      self, invalid_input, error_msg):
    """Tests that the shape exceptions are properly raised."""
    with self.assertRaisesRegexp(ValueError, error_msg):
      utils.extract_unique_edges_from_triangular_mesh(invalid_input)


if __name__ == "__main__":
  test_case.main()
