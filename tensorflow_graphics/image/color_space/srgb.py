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
"""This module implements Tensorflow sRGB color space utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import tensorflow as tf
from tensorflow_graphics.util import asserts

# Conversion constants following the naming convention from the 'theory of the
# transformation' section at https://en.wikipedia.org/wiki/SRGB.
_A = 0.055
_PHI = 12.92
_K0 = 0.04045


def _check_shape(tensor):
  """Checks if a Tensor has rank >=1 and last dimension 3."""
  if len(tensor.get_shape()) < 1:
    raise ValueError("Input Tensor must be of rank >= 1.")
  if tensor.get_shape()[-1] != 3:
    raise ValueError("Input Tensor must have last dimension equal to 3.")


def to_linear(srgb, gamma=2.4, name=None):
  """Converts sRGB colors to linear colors.

  Args:
    srgb: A Tensor of shape [dim_1, ..., dim_n, 3] with the last dimension
      holding sRGB values.
    gamma: A float gamma value to use for the conversion.
    name: A name for this op. Defaults to "srgb_to_linear".

  Raises:
    ValueError: If `srgb` has rank < 1 or has its last dimension not equal to 3.

  Returns:
    A Tensor of RGB values in linear color space.
  """
  with tf.compat.v1.name_scope(name, "srgb_to_linear", [srgb]):
    srgb = tf.convert_to_tensor(value=srgb)
    _check_shape(srgb)
    asserts.assert_all_in_range(srgb, 0., 1.)
    return tf.where(srgb <= _K0, srgb / _PHI, ((srgb + _A) / (1 + _A))**gamma)


def from_linear(linear, gamma=2.4, name=None):
  """Converts linear colors to sRGB colors.

  Args:
    linear: A Tensor of shape [dim_1, ..., dim_n, 3] with the last dimension
      holding RGB values in the range [0, 1] in linear color space.
    gamma: A float gamma value to use for the conversion.
    name: A name for this op. Defaults to "srgb_from_linear".

  Raises:
    ValueError: If `linear` has rank < 1 or has its last dimension not equal to
      3.

  Returns:
    A Tensor of sRGB values.
  """
  with tf.compat.v1.name_scope(name, "srgb_from_linear", [linear]):
    linear = tf.convert_to_tensor(value=linear)
    _check_shape(linear)
    asserts.assert_all_in_range(linear, 0., 1.)
    # Adds a small eps to avoid nan gradients from the second branch of
    # tf.where.
    linear += sys.float_info.epsilon
    return tf.where(linear <= _K0 / _PHI, linear * _PHI,
                    (1 + _A) * (linear**(1 / gamma)) - _A)


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
