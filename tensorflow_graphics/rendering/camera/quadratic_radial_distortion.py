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
"""Quadratic radial lens distortion and un-distortion functions.

Given a vector in homogeneous coordinates, (x/z, y/z, 1), we define
r^2 = (x/z)^2 + (y/z)^2. We use the simplest form of distortion function,
f(r) = 1 + k * r^2. The distorted vector is given by
(f(r) * x/z, f(r) * y/z, 1).

To apply the undistortion, we need the inverse of f(r), g = f^{-1}. In this
library we use the approximate formula for the undistortion function given here
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4934233/, and refine the solution
using Newton-Raphson iterations (https://en.wikipedia.org/wiki/Newtons_method).

Restricting the distortion function to quadratic form allows to easily detect
the cases where r goes beyond the monotonically-increasing range of f (which we
refer to as overflow).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api


def distortion_factor(squared_radius, distortion_coefficient, name=None):
  """Calculates a quadratic distortion factor given squared radii.

  Given a vector describing a location in camera space in homogeneous
  coordinates, (x/z, y/z, 1), squared_radius is r^2 = (x/z)^2 + (y/z)^2.
  distortion_factor multiplies x/z and y/z to obtain the distorted coordinates.
  In this function, distortion_factor is given by
  1.0 + `distortion_coefficient * squared_radius`.

  Args:
    squared_radius: n-D tensor of shape `[D1, ..., Dn]`, containing (x/z)^2 +
      (y/z)^2. We use squared_radius rather than the radius itself to avoid an
      unnecessary sqrt, which may introduce gradient singularities. The
      non-negativity of squared_radius is only enforced in debug mode.
    distortion_coefficient: m-D tensor of shape `[D1, ..., Dm]`, 0 <= m < n. The
      shape of distortion_coefficient will be appended by ones until the rank
      equals that of squared_radius.
    name: A name for this op. Defaults to
      "quadratic_radial_distortion_distortion_factor".

  Returns:
    distortion_factor: A tensor of shape `[D1, ..., Dn]`, the correction factor
      that should multiply the projective coordinates (x/z) and (y/z) to apply
      the distortion.
    overflow_mask: A boolean tensor of shape `[D1, ..., Dn]`, True where
      squared_radius is beyond the range where the distortion function is
      monotonically increasing. Wherever overflow_mask is True,
      distortion_factor's value is meaningless.
  """
  with tf.compat.v1.name_scope(name,
                               'quadratic_radial_distortion_distortion_factor',
                               [squared_radius, distortion_coefficient]):
    squared_radius = tf.convert_to_tensor(value=squared_radius)
    distortion_coefficient = tf.convert_to_tensor(value=distortion_coefficient)
    squared_radius = asserts.assert_all_above(
        squared_radius, 0.0, open_bound=False)
    distortion_coefficient = _expand_dims_to_match_rank(distortion_coefficient,
                                                        squared_radius)
    distortion_coefficient_times_squared_radius = (
        distortion_coefficient * squared_radius)
    distortion_factor_ = 1.0 + distortion_coefficient_times_squared_radius
    # This condition needs to hold for the distortion to be monotomnically
    # increasing, as can be derived by differentiating it.
    overflow_mask = tf.less(
        1.0 + 3.0 * distortion_coefficient_times_squared_radius, 0.0)
    return distortion_factor_, overflow_mask


def undistortion_factor(distorted_squared_radius,
                        distortion_coefficient,
                        num_iterations=5,
                        name=None):
  """Calculates the inverse quadratic distortion function given squared radii.

  Given a vector describing a location in camera space in homogeneous
  coordinates, (x/z, y/z, 1), after distortion has been applied, these become
  (x'/z, y'/z, 1). distorted_squared_radius is (x'/z)^2 + (y'/z)^2.
  undistortion_factor multiplies x'/z and y'/z to obtain the undistorted
  projective coordinates x/z and y/z.
  The undustortion factor in this function is derived from a quadratic.
  distortion function, where the distortion factor equals
  1.0 + `distortion_coefficient` * `squared_radius`.

  Args:
    distorted_squared_radius: n-D tensor of shape `[D1, ..., Dn]`. containing
      the value of  projective coordinates (x/z)^2 + (y/z)^2  For each pixel it
      contains the squared distance of that pixel to the center of the image
      plane. We use `distorted_squared_radius` rather than the distorted radius
      itself to avoid an unnecessary sqrt, which may introduce gradient
      singularities. The non-negativity of `distorted_squared_radius` is only
      enforced in debug mode.
    distortion_coefficient: m-D tensor of shape `[D1, ..., Dm]`, 0 <= m < n. The
      shape of distortion_coefficient will be appended by ones until the rank
      equals that of `distorted_squared_radius`.
    num_iterations: Number of Newton-Raphson iterations to calculate the inverse
      distprtion function. Defaults to 5, which is on the high-accuracy side.
    name: A name for this op. Defaults to
      "quadratic_radial_distortion_undistortion_factor".

  Returns:
    undistortion_factor: A tensor of shape `[D1, ..., Dn]` containing the
      correction factor that should multiply the distorted  projective
      coordinates (x'/z) and (y'/z) to obtain the undistorted ones.
    overflow_mask: A boolean tensor oof shape `[D1, ..., Dn]`, True where
      distorted_squared_radius is beyond the range where the distortion function
      is monotonically increasing. Wherever overflow_mask is True,
      undistortion_factor's value is meaningless.

  """
  with tf.compat.v1.name_scope(
      name, 'quadratic_radial_distortion_undistortion_factor',
      [distorted_squared_radius, distortion_coefficient]):
    distortion_coefficient = _expand_dims_to_match_rank(
        distortion_coefficient, distorted_squared_radius)
    distorted_squared_radius = tf.convert_to_tensor(
        value=distorted_squared_radius)
    distortion_coefficient = tf.convert_to_tensor(value=distortion_coefficient)
    distorted_squared_radius = asserts.assert_all_above(
        distorted_squared_radius, 0.0, open_bound=False)
    # For a distortion function of r' = (1 + ar^2)r, with a negative a, the
    # maximum r until which r'(r) is monotonically increasing is r^2 = -1/(3a).
    # At that value, r'^2 = -4 / (27a). Therefore the overflow condition for r'
    # is ar'^2 +(4/27.0) < 0. For a positive a it never holds, as it should,
    # because then r' is monotonic in r everywhere and thus never overflows.
    distortion_coefficient_times_distorted_squared_radius = (
        distortion_coefficient * distorted_squared_radius)
    overflow_mask = tf.less(
        4.0 / 27.0 + distortion_coefficient_times_distorted_squared_radius, 0.0)

    # Newton-raphson iterations. The expression below is obtained from
    # algebrically simplifying the Newton-Raphson formula
    # (https://en.wikipedia.org/wiki/Newtons_method)
    # We initialize with the approximate formula for the undistortion function
    # given here https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4934233/.
    undistortion_factor_ = (
        1.0 - distortion_coefficient_times_distorted_squared_radius)
    for _ in range(num_iterations):
      two_thirds_undistortion_factor = 2.0 * undistortion_factor_ / 3.0
      undistortion_factor_ = (1.0 - two_thirds_undistortion_factor) / (
          1.0 + 3.0 * distortion_coefficient_times_distorted_squared_radius *
          undistortion_factor_ *
          undistortion_factor_) + two_thirds_undistortion_factor
    return undistortion_factor_, overflow_mask


def _expand_dims_to_match_rank(tensor1, tensor2):
  """Appends ones to tensor1's shape until the rank matches that of tensor2."""
  with tf.compat.v1.name_scope('ExpandDimsToMatchRank', None,
                               [tensor1, tensor2]):
    tensor1_shape = tf.shape(input=tensor1)
    tensor2_rank = tf.rank(tensor2)
    padding_size = asserts.assert_all_above(
        tensor2_rank - tf.size(input=tensor1_shape), 0, open_bound=False)
    tensor1_shape = tf.concat(
        (tensor1_shape, tf.ones((padding_size,), dtype=tf.int32)), axis=0)
    return tf.reshape(tensor1, tensor1_shape)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
