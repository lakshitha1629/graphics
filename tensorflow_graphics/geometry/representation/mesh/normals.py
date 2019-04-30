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
"""Tensorflow utility functions to compute normals on meshes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.geometry.representation import triangle
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def gather_faces(vertices, indices, name=None):
  """Gather corresponding vertices for each face.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vertices: A tensor of shape `[A1, ..., An, V, D]`, where `V` is the number
      of vertices and `D` the dimensionality of each vertex. The rank of this
      tensor should be at least 2.
    indices: A tensor of shape `[A1, ..., An, F, M]`, where `F` is the number of
      faces, and `M` is the number of vertices per face. The rank of this tensor
      should be at least 2.
    name: A name for this op. Defaults to "normals_gather_faces".

  Returns:
    A tensor of shape `[A1, ..., An, F, M, D]` containing the vertices of each
    face.

  Raises:
    ValueError: If the shape of `vertices` or `indices` is not supported.
  """
  with tf.compat.v1.name_scope(name, "normals_gather_faces",
                               [vertices, indices]):
    vertices = tf.convert_to_tensor(value=vertices)
    indices = tf.convert_to_tensor(value=indices)

    shape.check_static(
        tensor=vertices, tensor_name="vertices", has_rank_greater_than=1)
    shape.check_static(
        tensor=indices, tensor_name="indices", has_rank_greater_than=1)
    shape.compare_batch_dimensions(
        tensors=(vertices, indices),
        last_axes=(-3, -3),
        broadcast_compatible=False)

    if hasattr(tf, "batch_gather"):
      expanded_vertices = tf.expand_dims(vertices, axis=-3)
      broadcasted_shape = tf.concat([tf.shape(input=indices)[:-1],
                                     tf.shape(input=vertices)[-2:]],
                                    axis=-1)
      broadcasted_vertices = tf.broadcast_to(
          expanded_vertices,
          broadcasted_shape)
      return tf.batch_gather(broadcasted_vertices, indices)
    else:
      return tf.gather(
          vertices, indices, axis=-2, batch_dims=indices.shape.ndims - 2)


def face_normals(faces, clockwise=True, normalize=True, name=None):
  """Computes face normals for meshes.

  This function supports planar convex polygon faces. Note that for
  non-triangular faces, this function uses the first 3 vertices of each
  face to calculate the face normal.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    faces: A tensor of shape `[A1, ..., An, M, 3]`, which stores vertices
      positions of each face, where M is the number of vertices of each face.
      The rank of this tensor should be at least 2.
    clockwise: Winding order to determine front-facing faces. The order of
      vertices should be either clockwise or counterclockwise.
    normalize: A `bool` defining whether output normals are normalized.
    name: A name for this op. Defaults to "normals_face_normals".

  Returns:
    A tensor of shape `[A1, ..., An, 3]` containing the face normals.

  Raises:
    ValueError: If the shape of `vertices`, `faces` is not supported.
  """
  with tf.compat.v1.name_scope(name, "normals_face_normals", [faces]):
    faces = tf.convert_to_tensor(value=faces)

    shape.check_static(
        tensor=faces,
        tensor_name="faces",
        has_rank_greater_than=1,
        has_dim_equals=(-1, 3),
        has_dim_greater_than=(-2, 2))

    vertices = tf.unstack(faces, axis=-2)
    vertices = vertices[:3]
    return triangle.normal(*vertices, clockwise=clockwise, normalize=normalize)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
