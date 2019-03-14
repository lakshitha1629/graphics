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
"""Tensorflow grid utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import tensorflow as tf


def _grid(starts, stops, nums):
  """Generates a M-D uniform axis-aligned grid.

  Note that the gradient of tf.lin_space and tf.meshgrid are currently not
  defined, it is currently not possible to backpropagate through the parameters
  of this op.

  Args:
    starts: 1-D tensor of shape `[M]` representing the start points for each
      dimension.
    stops: 1-D tensor of shape `[M]` representing the end points for each
      dimension.
    nums: 1-D tensor of shape `[M]` representing the number of subdivisions for
      each dimension.

  Returns:
    (M+1)-D tensor of shape `[?, ..., ?, M]`.
  """
  params = [tf.unstack(tensor) for tensor in [starts, stops, nums]]
  layout = [tf.linspace(*param) for param in zip(*params)]
  return tf.stack(tf.meshgrid(*layout, indexing="ij"), axis=-1)


def generate(starts, stops, nums, name=None):
  """Generates a M-D uniform axis-aligned grid.

  Note that the gradient of tf.lin_space and tf.meshgrid are currently not
  defined, it is currently not possible to backpropagate through the parameters
  of this op.

  Args:
    starts: 1-D or 2-D tensor of shape `[M]` or `[B, M]` representing the start
      points for each dimension.
    stops: 1-D or 2-D tensor of shape `[M]` or `[B, M]` representing the end
      points for each dimension.
    nums: 1-D tensor of shape `[M]` representing the number of subdivisions for
      each dimension.
    name: A name for this op. Defaults to "grid_generate".

  Returns:
    (M+1)-D or (M+2)-D tensor of shape `[?, ..., ?, M]` or [B, ?, ..., ?, M]`.
    Please refer to the example for more details.

  Raises:
    ValueError: If the shape of `starts`, `stops`, or 'nums' is not supported.

  Example:
    >>> generate((-1.0, -2.0), (1.0, 2.0), (3, 5))
    [[[-1. -2.]
      [-1. -1.]
      [-1.  0.]
      [-1.  1.]
      [-1.  2.]]
     [[ 0. -2.]
      [ 0. -1.]
      [ 0.  0.]
      [ 0.  1.]
      [ 0.  2.]]
     [[ 1. -2.]
      [ 1. -1.]
      [ 1.  0.]
      [ 1.  1.]
      [ 1.  2.]]]
    Generates a 3x5 2d grid from -1.0 to 1.0 with 3 subdivisions for the x axis
    and from -2.0 to 2.0 with 5 subdivisions for the y axis. This lead to a
    tensor of shape (3, 5, 2).
  """
  with tf.compat.v1.name_scope(name, "grid_generate", [starts, stops, nums]):
    starts = tf.convert_to_tensor(value=starts)
    stops = tf.convert_to_tensor(value=stops)
    nums = tf.convert_to_tensor(value=nums)
    starts_shape = starts.shape.as_list()
    stops_shape = stops.shape.as_list()
    nums_shape = nums.shape.as_list()
    if len(starts_shape) not in [1, 2]:
      raise ValueError("'starts' must be a 1-D or 2-D tensor.")
    if len(stops_shape) not in [1, 2]:
      raise ValueError("'stops' must be a 1-D or 2-D tensor.")
    if len(nums_shape) != 1:
      raise ValueError("'nums' must be a 1-D tensor.")
    if starts_shape != stops_shape:
      raise ValueError("'starts' and 'stops' must have the same shape.")
    if starts_shape[-1] != nums_shape[-1] or stops_shape[-1] != nums_shape[-1]:
      raise ValueError(
          "'starts', 'stops', and 'nums' must have the same last dimension.")

    if len(starts_shape) == 1:
      return _grid(starts, stops, nums)
    else:
      return tf.stack([
          _grid(starts, stops, nums)
          for starts, stops in zip(tf.unstack(starts), tf.unstack(stops))
      ])


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
