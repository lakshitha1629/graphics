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
"""This module implements lambertian reflectance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import math as m
import sys

import tensorflow as tf

from tensorflow_graphics.util import asserts


def brdf(albedo, name=None):
  """Evaluates the brdf of a Lambertian surface.

  Args:
    albedo: N-D tensor of shape `[?, ..., ?, 3]` with values in [0,1].
    name: A name for this op. Defaults to "lambertian_brdf".

  Returns:
      N-D tensor of shape `[?, ..., ?, 3]`.

  Raises:
    ValueError: if the shape of `albedo` is not supported.
    InvalidArgumentError: if at least an element of `albedo` is outside of
    [0,1].
  """
  with tf.compat.v1.name_scope(name, "lambertian_brdf", [albedo]):
    # Conversion to tensors
    albedo = tf.convert_to_tensor(value=albedo)
    shape = albedo.shape.as_list()
    if shape[-1] != 3:
      raise ValueError("'albedo' must have 3 dimensions.")
    albedo = asserts.assert_all_in_range(albedo, 0.0, 1.0, open_bounds=False)
    return albedo / m.pi


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
