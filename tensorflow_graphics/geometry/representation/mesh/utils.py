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
"""This module implements utility functions for meshes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow_graphics.util import export_api


def extract_unique_edges_from_triangular_mesh(faces):
  """Extracts all the unique edges using the faces of a mesh.

  Args:
    faces: A numpy.ndarray of shape [T, 3], where T is the number of triangular
      faces in the mesh. Each entry in this array describes the index of a
      vertex in the mesh.

  Returns:
    A numpy.ndarray of shape [E, 2], where E is the number of unique edges in
    the mesh.

  Raises:
    ValueError: If `faces` is not a numpy.ndarray or if its shape is not
      supported.
  """
  if not isinstance(faces, np.ndarray):
    raise ValueError("'faces' must be a numpy.ndarray.")
  faces_shape = faces.shape
  faces_rank = len(faces_shape)
  if faces_rank != 2:
    raise ValueError(
        "'faces' must have a rank equal to 2, but it has rank {} and shape {}."
        .format(faces_rank, faces_shape))
  if faces_shape[1] != 3:
    raise ValueError(
        "'faces' must have exactly 3 dimensions in the last axis, but it has {}"
        " dimensions and is of shape {}."
        .format(faces_shape[1], faces_shape))

  edges = []
  for face in faces:
    edges += [
        tuple(sorted((face[vertex_index], face[(vertex_index + 1) % 3])))
        for vertex_index in range(3)
    ]
  return np.array(list(set(edges)))


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
