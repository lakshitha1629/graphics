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
"""Tests for normals."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.representation.mesh import normals
from tensorflow_graphics.util import test_case


class MeshTest(test_case.TestCase):

  @parameterized.parameters(
      (((None, 3), (None, 3)), (tf.float32, tf.int32)),
      (((3, 6, 3), (3, 5, 4)), (tf.float32, tf.int32)),
  )
  def test_gather_faces_exception_not_raised(self, shapes, dtypes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(normals.gather_faces, shapes, dtypes)

  @parameterized.parameters(
      ("Not all batch dimensions are identical", (3, 5, 4, 4), (1, 2, 4, 4)),
      ("Not all batch dimensions are identical", (5, 4, 4), (1, 2, 4, 4)),
      ("Not all batch dimensions are identical", (3, 5, 4, 4), (2, 4, 4)),
      ("vertices must have a rank greater than 1", (4,), (1, 2, 4, 4)),
      ("indices must have a rank greater than 1", (3, 5, 4, 4), (4,)),
  )
  def test_gather_faces_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(normals.gather_faces, error_msg, shapes)

  def test_gather_faces_jacobian_random(self):
    """Test the Jacobian of the face extraction function."""
    tensor_size = np.random.randint(2, 5)
    tensor_shape = np.random.randint(1, 5, size=tensor_size).tolist()
    vertex_init = np.random.random(size=tensor_shape)
    indices_init = np.random.randint(0, tensor_shape[-2], size=tensor_shape)
    vertex_tensor = tf.convert_to_tensor(value=vertex_init)
    indices_tensor = tf.convert_to_tensor(value=indices_init)

    y = normals.gather_faces(vertex_tensor, indices_tensor)

    self.assert_jacobian_is_correct(vertex_tensor, vertex_init, y)

  @parameterized.parameters(
      ((((0.,), (1.,)), ((1, 0),)), ((((1.,), (0.,)),),)),
      ((((0., 1.), (2., 3.)), ((1, 0),)), ((((2., 3.), (0., 1.)),),)),
      ((((0., 1., 2.), (3., 4., 5.)), ((1, 0),)), ((((3., 4., 5.),
                                                     (0., 1., 2.)),),)),
  )
  def test_gather_faces_preset(self, test_inputs, test_outputs):
    """Tests the extraction of mesh faces."""
    self.assert_output_is_correct(
        normals.gather_faces, test_inputs, test_outputs, tile=False)

  def test_gather_faces_random(self):
    """Tests the extraction of mesh faces."""
    tensor_size = np.random.randint(3, 5)
    tensor_shape = np.random.randint(1, 5, size=tensor_size).tolist()
    vertices = np.random.random(size=tensor_shape)
    indices = np.arange(tensor_shape[-2])
    indices = indices.reshape([1] * (tensor_size - 1) + [-1])
    indices = np.tile(indices, tensor_shape[:-2] + [1, 1])
    expected = np.expand_dims(vertices, -3)

    self.assertAllClose(
        normals.gather_faces(vertices, indices), expected, rtol=1e-3)

  @parameterized.parameters(
      (((None, 4, 3),), (tf.float32,)),
      (((4, 3),), (tf.float32,)),
      (((3, 4, 3),), (tf.float32,)),
  )
  def test_face_normals_exception_not_raised(self, shapes, dtypes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(normals.face_normals, shapes, dtypes)

  @parameterized.parameters(
      ("faces must have a rank greater than 1.", (3,)),
      ("faces must have greater than 2 dimensions in axis -2", (2, 3)),
      ("faces must have exactly 3 dimensions in axis -1.", (5, 2)),
  )
  def test_face_normals_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(normals.face_normals, error_msg, shapes)

  def test_face_normals_jacobian_random(self):
    """Test the Jacobian of the face normals function."""
    tensor_vertex_size = np.random.randint(1, 3)
    tensor_out_shape = np.random.randint(1, 5, size=tensor_vertex_size)
    tensor_out_shape = tensor_out_shape.tolist()
    tensor_vertex_shape = list(tensor_out_shape)
    tensor_vertex_shape[-1] *= 3
    tensor_index_shape = tensor_out_shape[-1]
    vertex_init = np.random.random(size=tensor_vertex_shape + [3])
    index_init = np.arange(tensor_vertex_shape[-1])
    np.random.shuffle(index_init)
    index_init = np.reshape(index_init, newshape=[1] * (tensor_vertex_size - 1) \
                            + [tensor_index_shape, 3])
    index_init = np.tile(index_init, tensor_vertex_shape[:-1] + [1, 1])
    vertex_tensor = tf.convert_to_tensor(value=vertex_init)
    index_tensor = tf.convert_to_tensor(value=index_init)

    face_tensor = normals.gather_faces(vertex_tensor, index_tensor)
    y = normals.face_normals(face_tensor)

    self.assert_jacobian_is_correct(vertex_tensor, vertex_init, y)

  @parameterized.parameters(
      ((((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), ((0, 1, 2),)),
       (((0., 0., 1.),),)),
      ((((0., 0., 0.), (0., 0., 1.), (1., 0., 0.)), ((0, 1, 2),)),
       (((0., 1., 0.),),)),
      ((((0., 0., 0.), (0., 1., 0.), (0., 0., 1.)), ((0, 1, 2),)),
       (((1., 0., 0.),),)),
      ((((0., -2., -2.), (0, -2., 2.), (0., 2., 2.), (0., 2., -2.)),
        ((0, 1, 2, 3),)), (((-1., 0., 0.),),)),
  )
  def test_face_normals_preset(self, test_inputs, test_outputs):
    """Tests the computation of mesh face normals."""
    faces = normals.gather_faces(*test_inputs[:2])
    test_inputs = [faces] + list(test_inputs[2:])

    self.assert_output_is_correct(
        normals.face_normals, test_inputs, test_outputs, tile=False)

  def test_face_normals_random(self):
    """Tests the computation of mesh face normals in each axis."""
    tensor_vertex_size = np.random.randint(1, 3)
    tensor_out_shape = np.random.randint(1, 5, size=tensor_vertex_size)
    tensor_out_shape = tensor_out_shape.tolist()
    tensor_vertex_shape = list(tensor_out_shape)
    tensor_vertex_shape[-1] *= 3
    tensor_index_shape = tensor_out_shape[-1]

    for i in range(3):
      vertices = np.random.random(size=tensor_vertex_shape + [3])
      indices = np.arange(tensor_vertex_shape[-1])
      np.random.shuffle(indices)
      indices = np.reshape(indices,
                           newshape=[1] * (tensor_vertex_size - 1) \
                           + [tensor_index_shape, 3])
      indices = np.tile(indices, tensor_vertex_shape[:-1] + [1, 1])
      vertices[..., i] = 0.
      expected = np.zeros(shape=tensor_out_shape + [3], dtype=vertices.dtype)
      expected[..., i] = 1.
      faces = normals.gather_faces(vertices, indices)

      self.assertAllClose(
          tf.abs(normals.face_normals(faces)), expected, rtol=1e-3)


if __name__ == "__main__":
  test_case.main()
