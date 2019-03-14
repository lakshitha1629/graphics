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
"""Tensorflow triangle utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import tensorflow as tf

from tensorflow_graphics.geometry import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import shape


def normal(v0, v1, v2, clockwise=False, name=None):
  """Computes face normals (triangles).

  Args:
    v0: N-D tensor of shape `[?, ..., ?, 3]`.
    v1: N-D tensor of shape `[?, ..., ?, 3]`.
    v2: N-D tensor of shape `[?, ..., ?, 3]`.
    clockwise: Winding order to determine front-facing triangles.
    name: A name for this op. Defaults to "triangle_normal".

  Returns:
    N-D tensor of shape `[?, ..., ?, 3]` containing the normal.

  Raises:
    ValueError: If the shape of `v0`, `v1`, or `v2` is not supported.
  """
  with tf.compat.v1.name_scope(name, "triangle_normal", [v0, v1, v2]):
    v0 = tf.convert_to_tensor(value=v0)
    v1 = tf.convert_to_tensor(value=v1)
    v2 = tf.convert_to_tensor(value=v2)
    if not shape.is_broadcast_compatible(v0.shape, v1.shape):
      raise ValueError("'v0' and 'v1' should be broadcastable.")
    if not shape.is_broadcast_compatible(v0.shape, v2.shape):
      raise ValueError("'v0' and 'v2' should be broadcastable.")

    shape_v0 = v0.shape.as_list()
    shape_v1 = v1.shape.as_list()
    shape_v2 = v2.shape.as_list()
    if shape_v0[-1] != 3:
      raise ValueError("'v0' must have 3 dimensions.")
    if shape_v1[-1] != 3:
      raise ValueError("'v1' must have 3 dimensions.")
    if shape_v2[-1] != 3:
      raise ValueError("'v2' must have 3 dimensions.")

    n = vector.cross(v1 - v0, v2 - v0, axis=-1)
    n = asserts.assert_nonzero_norm(n)
    if not clockwise:
      n *= -1.0
    return tf.nn.l2_normalize(n, axis=-1)


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
