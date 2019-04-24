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
import numpy as np
import tensorflow as tf

from tensorflow_graphics.math.interpolation import bspline
from tensorflow_graphics.util import test_case


class BSplineTest(test_case.TestCase):

  @parameterized.parameters((0.0, (1.0,)), (1.0, (1.0,)))
  def test_constant_basis_boundary_values(self, position, weights):
    """Tests that basis functions of degree 0 return expected values."""
    self.assertAllClose(bspline._constant(position), weights)

  @parameterized.parameters((0.0, (1.0, 0.0)), (1.0, (0.0, 1.0)))
  def test_linear_basis_boundary_values(self, position, weights):
    """Tests that basis functions of degree 1 return expected values."""
    self.assertAllClose(bspline._linear(position), weights)

  @parameterized.parameters((0.0, (0.5, 0.5, 0.0)), (1.0, (0.0, 0.5, 0.5)))
  def test_quadratic_basis_boundary_values(self, position, weights):
    """Tests that basis functions of degree 2 return expected values."""
    self.assertAllClose(bspline._quadratic(position), weights)

  @parameterized.parameters((0.0, (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0, 0.0)),
                            (1.0, (0.0, 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)))
  def test_cubic_basis_boundary_values(self, position, weights):
    """Tests that basis functions of degree 3 return expected values."""
    self.assertAllClose(bspline._cubic(position), weights)

  @parameterized.parameters(
      (0.0, (1.0 / 24.0, 11.0 / 24.0, 11.0 / 24.0, 1.0 / 24.0, 0.0)),
      (1.0, (0.0, 1.0 / 24.0, 11.0 / 24.0, 11.0 / 24.0, 1.0 / 24.0)))
  def test_quartic_basis_boundary_values(self, position, weights):
    """Tests that basis functions of degree 4 return expected values."""
    self.assertAllClose(bspline._quartic(position), weights)

  @parameterized.parameters((((0.5,),), (((0.5, 0.5, 0.0),),)),
                            (((1.5,),), (((0.0, 0.5, 0.5),),)),
                            (((2.5,),), (((0.5, 0.0, 0.5),),)))
  def test_low_degree_knot_weights(self, position, weights):
    """Tests that knot weights are correct when degree < num_knots - 1."""
    self.assertAllClose(
        bspline.knot_weights(position, num_knots=3, degree=1, cyclical=True),
        weights)

    # pyformat: disable
  @parameterized.parameters(
      (((0.5,), (1.5,), (2.5,)),
       ((((0.5, 0.5),), ((0.5, 0.5),), ((0.5, 0.5),))),
       (((0,), (1,), (2,))), 1, True),
      ((0.0, 1.0), ((0.5, 0.5, 0.0), (0.0, 0.5, 0.5)), (0, 0), 2, False))
  # pyformat: enable
  def test_sparse_mode(self, positions, gt_weights, gt_shifts, degree,
                       cyclical):
    """Tests that sparse mode returns correct results."""
    weights, shifts = bspline.knot_weights(
        positions,
        num_knots=3,
        degree=degree,
        cyclical=cyclical,
        sparse_mode=True)
    self.assertAllClose(weights, gt_weights)
    self.assertAllClose(shifts, gt_shifts)

  @parameterized.parameters(
      (((0.5,), (1.5,), (2.5,)),
       ((((1.0 / 8.0, 0.75, 1.0 / 8.0),), ((1.0 / 8.0, 1.0 / 8.0, 0.75),),
         ((0.75, 1.0 / 8.0, 1.0 / 8.0),)))))
  def test_full_degree_knot_weights(self, positions, weights):
    """Tests that cyclical weights are correct when using max degree."""
    self.assertAllClose(
        bspline.knot_weights(positions, num_knots=3, degree=2, cyclical=True),
        weights)

  @parameterized.parameters((((0.0,), (0.25,), (0.5,), (0.75,)),))
  def test_full_degree_non_cyclical_knot_weights(self, positions):
    """Tests that noncyclical weights are correct when using max degree."""
    num_knots = 3
    degree = 2
    cyc_weights = bspline.knot_weights(positions, num_knots, degree, True)
    noncyc_weights = bspline.knot_weights(positions, num_knots, degree, False)
    self.assertAllClose(cyc_weights, noncyc_weights)

  @parameterized.parameters(1, 2, 3, 4)
  def test_positions_ranks(self, rank_positions):
    """Tests that different rank inputs work correctly."""
    shape = [2] * rank_positions
    positions = tf.constant(0.5, shape=shape)
    try:
      bspline.knot_weights(
          positions=positions, num_knots=3, degree=2, cyclical=True)
    except Exception as e:  # pylint: disable=broad-except
      self.fail("Exception raised: %s" % str(e))

  @parameterized.parameters(1, 2, 3, 4)
  def test_none_positions(self, rank_positions):
    """Tests that different rank inputs with None dimensions work correctly."""
    if tf.executing_eagerly():
      return
    shape = [None] * rank_positions
    shape_np = [2] * rank_positions
    positions = tf.compat.v1.placeholder(shape=shape, dtype=tf.float32)
    try:
      result = bspline.knot_weights(
          positions=positions, num_knots=3, degree=2, cyclical=True)
      with self.cached_session() as sess:
        sess.run(result, feed_dict={positions: np.ones(shape=shape_np)})
    except Exception as e:  # pylint: disable=broad-except
      self.fail("Exception raised: %s" % str(e))

  @parameterized.parameters(
      (((0.5,), (1.5,), (2.5,)), (((0.5, 1.5), (1.5, 1.5), (2.5, 3.5)),)))
  def test_num_knots_mismatch(self, weights, knots):
    """Tests that exception is raised when wrong number of knots is given."""
    with self.assertRaises(ValueError):
      bspline.interpolate_with_weights(knots, weights)

  @parameterized.parameters((1, 1), (2, 2), (3, 3), (3, 4))
  def test_large_degree(self, num_knots, degree):
    """Tests that exception is raised when given degree is too high."""
    positions = tf.constant(((0.5,), (1.5,), (2.5,)))
    with self.assertRaises(ValueError):
      bspline.knot_weights(positions, num_knots, degree, True)

  @parameterized.parameters((5), (6), (7))
  def test_max_degree(self, degree):
    """Tests that exception is raised when degree is > 4."""
    positions = tf.constant(((0.5,), (1.5,), (2.5,)))
    with self.assertRaises(NotImplementedError):
      bspline.knot_weights(positions, 10, degree, True)

  @parameterized.parameters(
      (((0.5,), (0.0,), (0.9,)), (((0.5, 1.5), (1.5, 1.5), (2.5, 3.5)),)))
  def test_interpolate_with_weights(self, positions, knots):
    """Tests that interpolate_with_weights works correctly."""
    degree = 1
    cyclical = False
    interp1 = bspline.interpolate(knots, positions, degree, cyclical)
    weights = bspline.knot_weights(positions, 2, degree, cyclical)
    interp2 = bspline.interpolate_with_weights(knots, weights)
    self.assertAllClose(interp1, interp2)


if __name__ == "__main__":
  test_case.main()
