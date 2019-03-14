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
"""Tests for as_conformal_as_possible."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.deformation_energies import as_conformal_as_possible
from tensorflow_graphics.transformation import quaternion
from tensorflow_graphics.util import test_case


class AsConformalAsPossibleTest(test_case.TestCase):

  def test_energy_identity(self):
    """Checks that energy evaluated between the rest pose and itself is zero."""
    number_vectices = np.random.randint(3, 10)
    batch_size = np.random.randint(3)
    batch_shape = np.random.randint(1, 10, size=(batch_size)).tolist()
    vertices_rest_pose = np.random.uniform(size=(number_vectices, 3))
    vertices_deformed_pose = tf.broadcast_to(
        vertices_rest_pose, shape=batch_shape + [number_vectices, 3])
    quaternions = quaternion.from_euler(
        np.zeros(shape=batch_shape + [number_vectices, 3]))
    num_edges = int(round(number_vectices / 2))
    edges = np.zeros(shape=(num_edges, 2), dtype=np.int32)
    edges[..., 0] = np.linspace(
        0, number_vectices / 2 - 1, num_edges, dtype=np.int32)
    edges[..., 1] = np.linspace(
        number_vectices / 2, number_vectices - 1, num_edges, dtype=np.int32)
    energy = as_conformal_as_possible.energy(
        vertices_rest_pose,
        vertices_deformed_pose,
        quaternions,
        edges,
        optimize_scales=False)
    self.assertAllClose(energy, tf.zeros_like(energy))

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_energy_jacobian_random(self):
    """Checks the correctness of the jacobian of energy."""
    number_vectices = np.random.randint(3, 10)
    batch_size = np.random.randint(3)
    batch_shape = np.random.randint(1, 10, size=(batch_size)).tolist()
    vertices_rest_pose_init = np.random.uniform(size=(number_vectices, 3))
    vertices_deformed_pose_init = np.random.uniform(size=batch_shape +
                                                    [number_vectices, 3])
    quaternions_init = np.random.uniform(size=batch_shape +
                                         [number_vectices, 4])
    num_edges = int(round(number_vectices / 2))
    edges = np.zeros(shape=(num_edges, 2), dtype=np.int32)
    edges[..., 0] = np.linspace(
        0, number_vectices / 2 - 1, num_edges, dtype=np.int32)
    edges[..., 1] = np.linspace(
        number_vectices / 2, number_vectices - 1, num_edges, dtype=np.int32)
    vertices_rest_pose = tf.convert_to_tensor(value=vertices_rest_pose_init)
    vertices_deformed_pose = tf.convert_to_tensor(
        value=vertices_deformed_pose_init)
    quaternions = tf.convert_to_tensor(value=quaternions_init)
    y = as_conformal_as_possible.energy(
        vertices_rest_pose,
        vertices_deformed_pose,
        quaternions,
        edges,
        optimize_scales=True)
    with self.subTest(name="rest_pose"):
      self.assert_jacobian_is_correct(vertices_rest_pose,
                                      vertices_rest_pose_init, y)
    with self.subTest(name="deformed_pose"):
      self.assert_jacobian_is_correct(vertices_deformed_pose,
                                      vertices_deformed_pose_init, y)
    with self.subTest(name="quaternions_scaled"):
      self.assert_jacobian_is_correct(quaternions, quaternions_init, y)
    with self.subTest(name="quaternions_normalized"):
      y = as_conformal_as_possible.energy(
          vertices_rest_pose,
          vertices_deformed_pose,
          quaternions,
          edges,
          optimize_scales=False)
      self.assert_jacobian_is_correct(quaternions, quaternions_init, y)

  @parameterized.parameters(
      ((1, 3), (1, 3), (1, 4), (1, 2), (1,), (1,)),
      ((1, 3), (None, 1, 3), (None, 1, 4), (1, 2), (1,), (1,)),
  )
  def test_energy_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        as_conformal_as_possible.energy, shapes,
        [tf.float32, tf.float32, tf.float32, tf.int32, tf.float32, tf.float32])

  def test_energy_preset(self):
    """Checks that energy returns the expected value."""
    vertices_rest_pose = np.array(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    vertices_deformed_pose = 2.0 * vertices_rest_pose
    quaternions = quaternion.from_euler(
        np.zeros(shape=vertices_deformed_pose.shape))
    edges = ((0, 1),)
    with self.subTest(name="all_weights_1"):
      energy = as_conformal_as_possible.energy(
          vertices_rest_pose, vertices_deformed_pose, quaternions, edges)
      gt = 4.0
      self.assertAllClose(energy, gt)
    with self.subTest(name="vertex_weights"):
      vertex_weights = np.array((2.0, 1.0))
      energy = as_conformal_as_possible.energy(
          vertices_rest_pose,
          vertices_deformed_pose,
          quaternions,
          edges,
          vertex_weight=vertex_weights)
      gt = 10.0
      self.assertAllClose(energy, gt)
    with self.subTest(name="edge_weights"):
      edge_weights = np.array((2.0,))
      energy = as_conformal_as_possible.energy(
          vertices_rest_pose,
          vertices_deformed_pose,
          quaternions,
          edges,
          edge_weight=edge_weights)
      gt = 16.0
      self.assertAllClose(energy, gt)

  @parameterized.parameters(
      ("'vertices_rest_pose' must be of length 2 and have the last dimension set to 3.",
       (1, 2), (1, 3), (1, 4), (1, 2), (1,), (1,)),
      ("'vertices_rest_pose' must be of length 2 and have the last dimension set to 3.",
       (2, 1, 2), (1, 3), (1, 4), (1, 2), (1,), (1,)),
      ("The last dimension of 'vertices_deformed_pose' must be 3.", (1, 3),
       (1, 2), (1, 4), (1, 2), (1,), (1,)),
      ("The number of vertices in 'vertices_rest_pose' and 'vertices_deformed_pose' must match.",
       (1, 3), (2, 3), (1, 4), (1, 2), (1,), (1,)),
      ("The last dimension of 'quaternions' must be 4.", (1, 3), (1, 3), (1, 5),
       (1, 2), (1,), (1,)),
      ("The number of vertices in 'vertices_rest_pose' and 'quaternions' must match.",
       (1, 3), (1, 3), (2, 4), (1, 2), (1,), (1,)),
      ("The shape of the batch in 'vertices_deformed_pose' and 'quaternions' must match.",
       (1, 3), (1, 3), (2, 1, 4), (1, 2), (1,), (1,)),
      ("'edges' must be of length 2 and have the last dimension set to 2.",
       (1, 3), (1, 3), (1, 4), (1, 3), (1,), (1,)),
      ("'edges' must be of length 2 and have the last dimension set to 2.",
       (1, 3), (1, 3), (1, 4), (2, 1, 2), (1,), (1,)),
      ("'vertex_weight' must be of dimension V, with V the number of vertices in the mesh.",
       (1, 3), (1, 3), (1, 4), (1, 2), (2,), (1,)),
      ("'edge_weight' must be of dimension E, with E the number of edges in the mesh.",
       (1, 3), (1, 3), (1, 4), (1, 2), (1,), (2,)),
  )
  def test_energy_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(as_conformal_as_possible.energy, error_msg,
                                    shape)


if __name__ == "__main__":
  test_case.main()
