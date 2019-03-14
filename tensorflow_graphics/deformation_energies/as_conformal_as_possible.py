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
"""This module implements TensorFlow As Rigid As Possible utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import tensorflow as tf

from tensorflow_graphics.geometry import vector
from tensorflow_graphics.transformation import quaternion


def energy(vertices_rest_pose,
           vertices_deformed_pose,
           quaternions,
           edges,
           vertex_weight=None,
           edge_weight=None,
           optimize_scales=True,
           name=None):
  """Estimates the As Conformal As Possible (ACAP) fitting energy.

  For a given mesh in rest pose, this function evaluates the ACAP fitting energy
  of a batch of deformed meshes. The vertex weights and edge weights are defined
  on the rest pose.

  Note:
    In the argument description below, V corresponds to the number of vertices
    in the mesh, E the number of edges, and [D1, ..., Dn] to the N-D shape
    of the batch.

  Args:
    vertices_rest_pose: 2-D tensor of shape `[V, 3]` containing the position of
      all the vertices of the mesh in rest pose.
    vertices_deformed_pose: (N+2)-D tensor of shape `[D1, ..., Dn, V, 3]`
    quaternions: (N+2)-D tensor of shape `[D1, ..., Dn, V, 4]`.
    edges: 2-D tensor of shape `[E, 2]` defining indices of vertices that are
      connected.
    vertex_weight: Optional 1-D tensor of shape `[V]` defining the weight
      associated with each vertex. Defaults to a tensor of ones.
    edge_weight: Optional 1-D tensor of shape `[E]` defining the weight of
      edges. Defaults to a tensor of ones.
    optimize_scales: Boolean indicating whether each vertex is associated with a
      scale factor or not.
    name: A name for this op. Defaults to "as_conformal_as_possible_energy".

  Returns:
    N-D tensor of shape `[D1, ..., Dn]` containing the ACAP energies.

  Raises:
    ValueError: if the shape of `vertices_rest_pose`, `vertices_deformed_pose`,
    `quaternions`, `edges`, `vertex_weight`, or `edge_weight` is not supported.
  """
  with tf.compat.v1.name_scope(name, "as_conformal_as_possible_energy", [
      vertices_rest_pose, vertices_deformed_pose, quaternions, edges,
      optimize_scales, vertex_weight, edge_weight
  ]):
    vertices_rest_pose = tf.convert_to_tensor(value=vertices_rest_pose)
    vertices_deformed_pose = tf.convert_to_tensor(value=vertices_deformed_pose)
    quaternions = tf.convert_to_tensor(value=quaternions)
    edges = tf.convert_to_tensor(value=edges)
    if vertex_weight is not None:
      vertex_weight = tf.convert_to_tensor(value=vertex_weight)
    if edge_weight is not None:
      edge_weight = tf.convert_to_tensor(value=edge_weight)
    if not optimize_scales:
      quaternions = quaternion.normalize(quaternions)
    # Checks the shape of inputs.
    shape_vertices_rest_pose = vertices_rest_pose.shape.as_list()
    if (len(shape_vertices_rest_pose) != 2 or
        shape_vertices_rest_pose[-1] != 3):
      raise ValueError(
          "'vertices_rest_pose' must be of length 2 and have the last dimension"
          " set to 3.")
    number_vertices = shape_vertices_rest_pose[-2]
    shape_vertices_deformed_pose = vertices_deformed_pose.shape.as_list()
    if shape_vertices_deformed_pose[-1] != 3:
      raise ValueError(
          "The last dimension of 'vertices_deformed_pose' must be 3.")
    if shape_vertices_deformed_pose[-2] != number_vertices:
      raise ValueError("The number of vertices in 'vertices_rest_pose' and "
                       "'vertices_deformed_pose' must match.")
    batch_shape = shape_vertices_deformed_pose[:-2]
    shape_quaternions = quaternions.shape.as_list()
    if shape_quaternions[-1] != 4:
      raise ValueError("The last dimension of 'quaternions' must be 4.")
    if shape_quaternions[-2] != number_vertices:
      raise ValueError(
          "The number of vertices in 'vertices_rest_pose' and 'quaternions' "
          "must match.")
    if shape_quaternions[:-2] != batch_shape:
      raise ValueError(
          "The shape of the batch in 'vertices_deformed_pose' and 'quaternions'"
          " must match.")
    shape_edges = edges.shape.as_list()
    if (len(shape_edges) != 2 or shape_edges[-1] != 2):
      raise ValueError(
          "'edges' must be of length 2 and have the last dimension set to 2.")
    if vertex_weight is not None:
      if vertex_weight.shape.as_list() != [number_vertices]:
        raise ValueError(
            "'vertex_weight' must be of dimension V, with V the number of "
            "vertices in the mesh.")
    if edge_weight is not None:
      if edge_weight.shape.as_list() != [shape_edges[0]]:
        raise ValueError(
            "'edge_weight' must be of dimension E, with E the number of edges "
            "in the mesh.")
    # Extracts the indices of vertices.
    indices_i, indices_j = tf.unstack(edges, axis=-1)
    # Extracts the vertices we need per term.
    vertices_i_rest = tf.gather(vertices_rest_pose, indices_i, axis=-2)
    vertices_j_rest = tf.gather(vertices_rest_pose, indices_j, axis=-2)
    vertices_i_deformed = tf.gather(vertices_deformed_pose, indices_i, axis=-2)
    vertices_j_deformed = tf.gather(vertices_deformed_pose, indices_j, axis=-2)
    # Extracts the weights we need per term.
    weights_shape = vertices_i_rest.shape.as_list()[-2]
    if vertex_weight is not None:
      weight_i = tf.gather(vertex_weight, indices_i)
      weight_j = tf.gather(vertex_weight, indices_j)
    else:
      weight_i = weight_j = tf.ones(
          weights_shape, dtype=vertices_rest_pose.dtype)
    weight_i = tf.expand_dims(weight_i, axis=-1)
    weight_j = tf.expand_dims(weight_j, axis=-1)
    if edge_weight is not None:
      weight_ij = edge_weight
    else:
      weight_ij = tf.ones(weights_shape, dtype=vertices_rest_pose.dtype)
    weight_ij = tf.expand_dims(weight_ij, axis=-1)
    # Extracts the rotation we need per term.
    quaternion_i = tf.gather(quaternions, indices_i, axis=-2)
    quaternion_j = tf.gather(quaternions, indices_j, axis=-2)
    # Computes the energy.
    deformed_ij = vertices_i_deformed - vertices_j_deformed
    rotated_rest_ij = quaternion.rotate((vertices_i_rest - vertices_j_rest),
                                        quaternion_i)
    energy_ij = weight_i * weight_ij * (deformed_ij - rotated_rest_ij)
    deformed_ji = vertices_j_deformed - vertices_i_deformed
    rotated_rest_ji = quaternion.rotate((vertices_j_rest - vertices_i_rest),
                                        quaternion_j)
    energy_ji = weight_j * weight_ij * (deformed_ji - rotated_rest_ji)
    average_energy_ij = tf.reduce_mean(
        input_tensor=vector.dot(energy_ij, energy_ij, keepdims=False), axis=-1)
    average_energy_ji = tf.reduce_mean(
        input_tensor=vector.dot(energy_ji, energy_ji, keepdims=False), axis=-1)
    return (average_energy_ij + average_energy_ji) / 2.0


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
