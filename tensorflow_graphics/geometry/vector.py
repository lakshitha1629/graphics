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
"""Tensorflow vector utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import tensorflow as tf

from tensorflow_graphics.util import asserts


def cross(u, v, axis=-1, name=None):
  """Computes the cross product between two tensors u and v along an axis.

  Args:
    u: Tensor of rank R and shape `[?, ..., 3, ..., ?]`.
    v: Tensor of rank R and shape `[?, ..., 3, ..., ?]`.
    axis: The axis to compute the cross product along.
    name: A name for this op. Defaults to "vector_cross".

  Returns:
    Tensor of rank R shape `[?, ..., 3, ..., ?]`.
  """
  with tf.compat.v1.name_scope(name, "vector_cross", [u, v]):
    u = tf.convert_to_tensor(value=u)
    v = tf.convert_to_tensor(value=v)
    shape_u = u.shape.as_list()
    shape_v = v.shape.as_list()
    if shape_u[axis] != 3:
      raise ValueError("'u' must have 3 dimensions at the given axis.")
    if shape_v[axis] != 3:
      raise ValueError("'v' must have 3 dimensions at the given axis.")

    ux, uy, uz = tf.unstack(u, axis=axis)
    vx, vy, vz = tf.unstack(v, axis=axis)
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    return tf.stack((nx, ny, nz), axis=axis)


def dot(u, v, axis=-1, keepdims=True, name=None):
  """Computes the dot product between two tensors u and v along an axis.

  Args:
    u: Tensor of rank R and shape `[?, ..., M, ..., ?]`.
    v: Tensor of rank R and shape `[?, ..., M, ..., ?]`.
    axis: The axis to compute the dot product along.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for this op. Defaults to "vector_dot".

  Returns:
    Tensor of rank R and shape `[?, ..., 1, ..., ?]`.
  """
  with tf.compat.v1.name_scope(name, "vector_dot", [u, v]):
    u = tf.convert_to_tensor(value=u)
    v = tf.convert_to_tensor(value=v)
    shape_u = u.shape.as_list()
    shape_v = v.shape.as_list()
    if shape_u[axis] != shape_v[axis]:
      raise ValueError(
          "'u' and 'v' must have the same dimension at the given axis.")
    return tf.reduce_sum(input_tensor=u * v, axis=axis, keepdims=keepdims)


def intersection_ray_sphere(sphere_center, sphere_radius, ray, point_on_ray):
  """Finds positions and surface normals where the sphere and the ray intersect.

  Args:
    sphere_center: N-D tensor of shape `[S, 3]`, with S a list of arbitrary
      shape.
    sphere_radius: N-D tensor of shape `[S, 1]` containing strictly positive
      values.
    ray: N-D tensor of shape `[R, 3]` containing normalized vectors. R is a list
      of arbitrary shape.
    point_on_ray: N-D tensor of shape `[R, 3]`.

  Returns:
    N-D tensor of shape `[?, ..., ?, 3]` containing the position of the
    intersection, and N-D tensor of shape `[?, ..., ?, 3]` the associated
    surface normal at that point. Both tensors contain NaNs when there is no
    intersections.

  Raises:
    ValueError: if the shape of `sphere_center`, `sphere_radius`, `ray` or
      `point_on_ray` is not supported.
    tf.errors.InvalidArgumentError: If `ray` is not normalized.
  """
  # Converts to tensors.
  sphere_center = tf.convert_to_tensor(value=sphere_center)
  sphere_radius = tf.convert_to_tensor(value=sphere_radius)
  ray = tf.convert_to_tensor(value=ray)
  point_on_ray = tf.convert_to_tensor(value=point_on_ray)
  # Checks the shape of the inputs.
  shape_sphere_center = sphere_center.shape.as_list()
  shape_sphere_radius = sphere_radius.shape.as_list()

  if shape_sphere_center[-1] != 3:
    raise ValueError("'sphere_center' must have 3 dimensions.")
  if shape_sphere_radius[-1] != 1:
    raise ValueError("'sphere_radius' must have 1 dimensions.")
  if shape_sphere_center[:-1] != shape_sphere_radius[:-1]:
    raise ValueError(
        "'sphere_center' and 'sphere_radius' must have the same shape except in the last dimension."
    )
  shape_ray = ray.shape.as_list()
  shape_point_on_ray = point_on_ray.shape.as_list()
  if shape_ray[-1] != 3:
    raise ValueError("'ray' must have 3 dimensions.")
  if shape_point_on_ray[-1] != 3:
    raise ValueError("'point_on_ray' must have 3 dimensions.")
  if shape_ray != shape_point_on_ray:
    raise ValueError("'ray' and 'point_on_ray' must have the same shape.")
  # Checks that 'ray' is normalized.
  sphere_radius = asserts.assert_all_above(sphere_radius, 0.0, open_bound=True)
  ray = asserts.assert_normalized(ray)
  # Computes the results.
  vector_sphere_center_to_point_on_ray = sphere_center - point_on_ray
  distance_sphere_center_to_point_on_ray = tf.norm(
      tensor=vector_sphere_center_to_point_on_ray, axis=-1, keepdims=True)
  distance_projection_sphere_center_on_ray = dot(
      vector_sphere_center_to_point_on_ray, ray)
  closest_distance_sphere_center_to_ray = tf.sqrt(
      tf.square(distance_sphere_center_to_point_on_ray) -
      tf.pow(distance_projection_sphere_center_on_ray, 2))
  p = tf.sqrt(
      tf.square(sphere_radius) -
      tf.square(closest_distance_sphere_center_to_ray))
  distances = tf.stack((distance_projection_sphere_center_on_ray - p,
                        distance_projection_sphere_center_on_ray + p))
  intersections_points = distances * ray + point_on_ray
  normals = tf.math.l2_normalize(intersections_points - sphere_center, axis=-1)
  return intersections_points, normals

# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
