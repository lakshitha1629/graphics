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
"""Tests for slerp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.interpolation import bspline
from tensorflow_graphics.util import test_case


class BSplineTest(test_case.TestCase):

  @parameterized.parameters((0.0, (1.0,)), (1.0, (1.0,)))
  def test_constant_basis_boundary_values(self, position, weights):
    self.assertAllClose(bspline._constant(position), weights)

  @parameterized.parameters((0.0, (1.0, 0.0)), (1.0, (0.0, 1.0)))
  def test_linear_basis_boundary_values(self, position, weights):
    self.assertAllClose(bspline._linear(position), weights)

  @parameterized.parameters((0.0, (0.5, 0.5, 0.0)), (1.0, (0.0, 0.5, 0.5)))
  def test_quadratic_basis_boundary_values(self, position, weights):
    self.assertAllClose(bspline._quadratic(position), weights)

  @parameterized.parameters((0.0, (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0, 0.0)),
                            (1.0, (0.0, 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)))
  def test_cubic_basis_boundary_values(self, position, weights):
    self.assertAllClose(bspline._cubic(position), weights)

  @parameterized.parameters(
      (0.0, (1.0 / 24.0, 11.0 / 24.0, 11.0 / 24.0, 1.0 / 24.0, 0.0)),
      (1.0, (0.0, 1.0 / 24.0, 11.0 / 24.0, 11.0 / 24.0, 1.0 / 24.0)))
  def test_quartic_basis_boundary_values(self, position, weights):
    self.assertAllClose(bspline._quartic(position), weights)

  @parameterized.parameters((((0.5,),), (((0.5, 0.5, 0.0),),)),
                            (((1.5,),), (((0.0, 0.5, 0.5),),)),
                            (((2.5,),), (((0.5, 0.0, 0.5),),)))
  def test_low_degree_knot_weights(self, position, weights):
    # Low degree means that degree < num_knots - 1, which results in zeros.
    self.assertAllClose(
        bspline.knot_weights(position, num_knots=3, degree=1, cyclical=True),
        weights)

  @parameterized.parameters((((0.5,), (1.5,), (2.5,)),
                             ((((0.5, 0.5),), ((0.5, 0.5),),
                               ((0.5, 0.5),))), (((0,), (1,), (2,)))))
  def test_sparse_mode(self, positions, gt_weights, gt_shifts):
    # Low degree means that degree < num_knots - 1, which results in zeros.
    weights, shifts = bspline.knot_weights(
        positions, num_knots=3, degree=1, cyclical=True, sparse_mode=True)
    self.assertAllClose(weights, gt_weights)
    self.assertAllClose(shifts, gt_shifts)

  @parameterized.parameters((((0.5,), (1.5,), (2.5,)),
                             ((((1.0 / 8.0, 0.75,
                                 1.0 / 8.0),), ((1.0 / 8.0, 1.0 / 8.0, 0.75),),
                               ((0.75, 1.0 / 8.0, 1.0 / 8.0),)))))
  def test_full_degree_knot_weights(self, positions, weights):
    # Full degree means that degree = num_knots - 1, with all weights nonzero.
    self.assertAllClose(
        bspline.knot_weights(positions, num_knots=3, degree=2, cyclical=True),
        weights)

  @parameterized.parameters((((0.0,), (0.25,), (0.5,), (0.75,)),))
  def test_full_degree_non_cyclical_knot_weights(self, positions):
    # Full degree means that degree = num_knots - 1, with all weights nonzero.
    num_knots = 3
    degree = 2
    cyc_weights = bspline.knot_weights(positions, num_knots, degree, True)
    noncyc_weights = bspline.knot_weights(positions, num_knots, degree, False)
    self.assertAllClose(cyc_weights, noncyc_weights)

  @parameterized.parameters(((0.5, 1.5, 2.5),))
  def test_positions(self, rank_1_positions):
    with self.assertRaises(ValueError):
      bspline.knot_weights(rank_1_positions, 3, 1, True)

  @parameterized.parameters((((0.5,), (1.5,), (2.5,)), (((0.5, 1.5), (1.5, 1.5),
                                                         (2.5, 3.5)),)))
  def test_num_knots_mismatch(self, weights, knots):
    with self.assertRaises(ValueError):
      bspline.interpolate_with_weights(knots, weights)

  @parameterized.parameters((1, 1), (2, 2), (3, 3), (3, 4))
  def test_large_degree(self, num_knots, degree):
    positions = tf.constant(((0.5,), (1.5,), (2.5,)))
    with self.assertRaises(ValueError):
      bspline.knot_weights(positions, num_knots, degree, True)

  @parameterized.parameters((5), (6), (7))
  def test_max_degree(self, degree):
    positions = tf.constant(((0.5,), (1.5,), (2.5,)))
    with self.assertRaises(NotImplementedError):
      bspline.knot_weights(positions, 10, degree, True)

  @parameterized.parameters((((0.5,), (0.0,), (0.9,)), (((0.5, 1.5), (1.5, 1.5),
                                                         (2.5, 3.5)),)))
  def test_interpolate_with_weights(self, positions, knots):
    degree = 1
    cyclical = False
    interp1 = bspline.interpolate(knots, positions, degree, cyclical)
    weights = bspline.knot_weights(positions, 2, degree, cyclical)
    interp2 = bspline.interpolate_with_weights(knots, weights)
    self.assertAllClose(interp1, interp2)


if __name__ == "__main__":
  test_case.main()
