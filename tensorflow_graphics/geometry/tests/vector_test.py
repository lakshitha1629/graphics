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
"""Tests for vector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry import vector
from tensorflow_graphics.transformation.tests import test_data as td
from tensorflow_graphics.util import test_case


class VectorTest(test_case.TestCase):

  @parameterized.parameters(
      ((None, 3), (None, 3)),)
  def test_cross_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(vector.cross, shapes)

  @parameterized.parameters(
      ("'u' must have 3 dimensions at the given axis.", (None,), (3,)),
      ("'v' must have 3 dimensions at the given axis.", (3,), (None,)),
  )
  def test_cross_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(vector.cross, error_msg, shapes)

  @parameterized.parameters(
      (td.AXIS_3D_0, td.AXIS_3D_0),
      (td.AXIS_3D_0, td.AXIS_3D_X),
      (td.AXIS_3D_0, td.AXIS_3D_Y),
      (td.AXIS_3D_0, td.AXIS_3D_Z),
      (td.AXIS_3D_X, td.AXIS_3D_X),
      (td.AXIS_3D_X, td.AXIS_3D_Y),
      (td.AXIS_3D_X, td.AXIS_3D_Z),
      (td.AXIS_3D_Y, td.AXIS_3D_X),
      (td.AXIS_3D_Y, td.AXIS_3D_Y),
      (td.AXIS_3D_Y, td.AXIS_3D_Z),
      (td.AXIS_3D_Z, td.AXIS_3D_X),
      (td.AXIS_3D_Z, td.AXIS_3D_Y),
      (td.AXIS_3D_Z, td.AXIS_3D_Z),
  )
  def test_cross_jacobian_preset(self, u_init, v_init):
    """Tests the Jacobian of the dot product."""
    u_tensor = tf.convert_to_tensor(value=u_init)
    v_tensor = tf.convert_to_tensor(value=v_init)
    y = vector.cross(u_tensor, v_tensor)
    self.assert_jacobian_is_correct(u_tensor, u_init, y)
    self.assert_jacobian_is_correct(v_tensor, v_init, y)

  def test_cross_jacobian_random(self):
    """Test the Jacobian of the dot product."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    u_init = np.random.random(size=tensor_shape + [3])
    v_init = np.random.random(size=tensor_shape + [3])
    u_tensor = tf.convert_to_tensor(value=u_init)
    v_tensor = tf.convert_to_tensor(value=v_init)
    y = vector.cross(u_tensor, v_tensor)
    self.assert_jacobian_is_correct(u_tensor, u_init, y)
    self.assert_jacobian_is_correct(v_tensor, v_init, y)

  @parameterized.parameters(
      ((td.AXIS_3D_0, td.AXIS_3D_0), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_0, td.AXIS_3D_X), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_0, td.AXIS_3D_Y), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_0, td.AXIS_3D_Z), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_X, td.AXIS_3D_X), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_X, td.AXIS_3D_Y), (td.AXIS_3D_Z,)),
      ((td.AXIS_3D_X, td.AXIS_3D_Z), (-td.AXIS_3D_Y,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_X), (-td.AXIS_3D_Z,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_Y), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_Z), (td.AXIS_3D_X,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_X), (td.AXIS_3D_Y,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_Y), (-td.AXIS_3D_X,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_Z), (td.AXIS_3D_0,)),
  )
  def test_cross_preset(self, test_inputs, test_outputs):
    """Tests the cross product of predefined axes."""
    self.assert_output_is_correct(vector.cross, test_inputs, test_outputs)

  def test_cross_random(self):
    """Tests the cross product function."""
    tensor_size = np.random.randint(1, 4)
    tensor_shape = np.random.randint(1, 10, size=tensor_size).tolist()
    axis = np.random.randint(tensor_size)
    tensor_shape[axis] = 3
    u = np.random.random(size=tensor_shape)
    v = np.random.random(size=tensor_shape)
    self.assertAllClose(
        vector.cross(u, v, axis=axis), np.cross(u, v, axis=axis))

  @parameterized.parameters(
      ((None,), (None,)),
      ((None, None), (None, None)),
  )
  def test_dot_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(vector.dot, shapes)

  @parameterized.parameters(
      ("'u' and 'v' must have the same dimension at the given axis.", (None, 1),
       (None, 2)),)
  def test_dot_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(vector.dot, error_msg, shapes)

  @parameterized.parameters(
      (td.AXIS_3D_0, td.AXIS_3D_0),
      (td.AXIS_3D_0, td.AXIS_3D_X),
      (td.AXIS_3D_0, td.AXIS_3D_Y),
      (td.AXIS_3D_0, td.AXIS_3D_Z),
      (td.AXIS_3D_X, td.AXIS_3D_X),
      (td.AXIS_3D_X, td.AXIS_3D_Y),
      (td.AXIS_3D_X, td.AXIS_3D_Z),
      (td.AXIS_3D_Y, td.AXIS_3D_X),
      (td.AXIS_3D_Y, td.AXIS_3D_Y),
      (td.AXIS_3D_Y, td.AXIS_3D_Z),
      (td.AXIS_3D_Z, td.AXIS_3D_X),
      (td.AXIS_3D_Z, td.AXIS_3D_Y),
      (td.AXIS_3D_Z, td.AXIS_3D_Z),
  )
  def test_dot_jacobian_preset(self, u_init, v_init):
    """Test the Jacobian of the dot product."""
    u_tensor = tf.convert_to_tensor(value=u_init)
    v_tensor = tf.convert_to_tensor(value=v_init)
    y = vector.dot(u_tensor, v_tensor)
    self.assert_jacobian_is_correct(u_tensor, u_init, y)
    self.assert_jacobian_is_correct(v_tensor, v_init, y)

  def test_dot_jacobian_random(self):
    """Test the Jacobian of the dot product."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    u_init = np.random.random(size=tensor_shape + [3])
    v_init = np.random.random(size=tensor_shape + [3])
    u_tensor = tf.convert_to_tensor(value=u_init)
    v_tensor = tf.convert_to_tensor(value=v_init)
    y = vector.dot(u_tensor, v_tensor)
    self.assert_jacobian_is_correct(u_tensor, u_init, y)
    self.assert_jacobian_is_correct(v_tensor, v_init, y)

  @parameterized.parameters(
      ((td.AXIS_3D_0, td.AXIS_3D_0), (0.,)),
      ((td.AXIS_3D_0, td.AXIS_3D_X), (0.,)),
      ((td.AXIS_3D_0, td.AXIS_3D_Y), (0.,)),
      ((td.AXIS_3D_0, td.AXIS_3D_Z), (0.,)),
      ((td.AXIS_3D_X, td.AXIS_3D_X), (1.,)),
      ((td.AXIS_3D_X, td.AXIS_3D_Y), (0.,)),
      ((td.AXIS_3D_X, td.AXIS_3D_Z), (0.,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_X), (0.,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_Y), (1.,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_Z), (0.,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_X), (0.,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_Y), (0.,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_Z), (1.,)),
  )
  def test_dot_preset(self, test_inputs, test_outputs):
    """Tests the dot product of predefined axes."""

    def func(u, v):
      return tf.squeeze(vector.dot(u, v), axis=-1)

    self.assert_output_is_correct(func, test_inputs, test_outputs)

  def test_dot_random(self):
    """Tests the dot product function."""
    tensor_size = np.random.randint(2, 4)
    tensor_shape = np.random.randint(1, 10, size=tensor_size).tolist()
    axis = np.random.randint(tensor_size)
    u = np.random.random(size=tensor_shape)
    v = np.random.random(size=tensor_shape)
    dot = tf.linalg.tensor_diag_part(tf.tensordot(u, v, axes=[[axis], [axis]]))
    dot = tf.expand_dims(dot, axis=axis)
    self.assertAllClose(vector.dot(u, v, axis=axis), dot)

  @parameterized.parameters(
      ("'sphere_center' must have 3 dimensions.", (2,), (1,), (3,), (3,)),
      ("'sphere_radius' must have 1 dimensions.", (3,), (2,), (3,), (3,)),
      ("'sphere_center' and 'sphere_radius' must have the same shape except in the last dimension.",
       (2, 3), (1,), (3,), (3,)),
      ("'ray' must have 3 dimensions.", (3,), (1,), (2,), (3,)),
      ("'point_on_ray' must have 3 dimensions.", (3,), (1,), (3,), (2,)),
      ("'ray' and 'point_on_ray' must have the same shape.", (3,), (1,), (3,),
       (2, 3)),
  )
  def test_intersection_ray_sphere_shape_raised(self, error_msg, *shapes):
    """tests that exceptions are raised when shapes are not supported."""
    self.assert_exception_is_raised(vector.intersection_ray_sphere, error_msg,
                                    shapes)

  @parameterized.parameters(
      ((3,), (1,), (3,), (3,)),
      ((None, 3), (None, 1), (None, 3), (None, 3)),
  )
  def test_intersection_ray_sphere_shape_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised on supported shapes."""
    self.assert_exception_is_not_raised(vector.intersection_ray_sphere, shapes)

  def test_intersection_ray_sphere_exception_raised(self):
    """Tests that exceptions are properly raised."""
    sphere_center = np.random.uniform(size=(3,))
    point_on_ray = np.random.uniform(size=(3,))
    with self.subTest(name="positive_radius"):
      sphere_radius = np.random.uniform(-1.0, 0.0, size=(1,))
      ray = np.random.uniform(size=(3,))
      ray /= np.linalg.norm(ray, axis=-1)
      with self.assertRaises(tf.errors.InvalidArgumentError):
        intersection, normal = vector.intersection_ray_sphere(
            sphere_center, sphere_radius, ray, point_on_ray)
        self.evaluate((intersection, normal))
    with self.subTest(name="normalized_ray"):
      ray = np.random.uniform(2.0, 3.0, size=(3,))
      sphere_radius = np.random.uniform(sys.float_info.epsilon, 1.0, size=(1,))
      with self.assertRaises(tf.errors.InvalidArgumentError):
        intersection, normal = vector.intersection_ray_sphere(
            sphere_center, sphere_radius, ray, point_on_ray)
        self.evaluate((intersection, normal))

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_intersection_ray_sphere_jacobian_random(self):
    """Test the Jacobian of the intersection_ray_sphere function."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    sphere_center_init = np.random.uniform(0.0, 1.0, size=tensor_shape + [3])
    sphere_radius_init = np.random.uniform(10.0, 11.0, size=tensor_shape + [1])
    ray_init = np.random.uniform(size=tensor_shape + [3])
    ray_init /= np.linalg.norm(ray_init, axis=-1, keepdims=True)
    point_on_ray_init = np.random.uniform(0.0, 1.0, size=tensor_shape + [3])
    # Convert to tensors.
    sphere_center = tf.convert_to_tensor(value=sphere_center_init)
    sphere_radius = tf.convert_to_tensor(value=sphere_radius_init)
    ray = tf.convert_to_tensor(value=ray_init)
    point_on_ray = tf.convert_to_tensor(value=point_on_ray_init)
    y_p, y_n = vector.intersection_ray_sphere(sphere_center, sphere_radius, ray,
                                              point_on_ray)
    self.assert_jacobian_is_correct(ray, ray_init, y_p)
    self.assert_jacobian_is_correct(ray, ray_init, y_n)
    self.assert_jacobian_is_correct(sphere_radius, sphere_radius_init, y_p)
    self.assert_jacobian_is_correct(sphere_radius, sphere_radius_init, y_n)
    self.assert_jacobian_is_correct(sphere_center, sphere_center_init, y_p)
    self.assert_jacobian_is_correct(sphere_center, sphere_center_init, y_n)
    self.assert_jacobian_is_correct(point_on_ray, point_on_ray_init, y_p)
    self.assert_jacobian_is_correct(point_on_ray, point_on_ray_init, y_n)

  @parameterized.parameters(
      (((0.0, 0.0, 3.0), (1.0,), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
       (((0.0, 0.0, 2.0), (0.0, 0.0, 4.0)), ((0.0, 0.0, -1.0),
                                             (0.0, 0.0, 1.0)))),
      (((0.0, 0.0, 3.0), (1.0,), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
       (((1.0, 0.0, 3.0), (1.0, 0.0, 3.0)), ((1.0, 0.0, 0.0),
                                             (1.0, 0.0, 0.0)))),
      (((0.0, 0.0, 3.0), (1.0,), (0.0, 0.0, 1.0), (2.0, 0.0, 0.0)),
       (((np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan)),
        ((np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan)))),
  )
  def test_intersection_ray_sphere_preset(self, test_inputs, test_outputs):
    self.assert_output_is_correct(
        vector.intersection_ray_sphere, test_inputs, test_outputs, tile=False)


if __name__ == "__main__":
  test_case.main()
