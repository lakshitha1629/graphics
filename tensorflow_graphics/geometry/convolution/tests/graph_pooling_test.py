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
"""Tests for google3.third_party.py.tensorflow_graphics.geometry.convolution.tests.graph_pooling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

import tensorflow_graphics.geometry.convolution.graph_pooling as gp
from tensorflow_graphics.geometry.convolution.tests import utils_test
from tensorflow_graphics.util import test_case


def _dense_to_sparse(data):
  """Convert a numpy array to a tf.SparseTensor."""
  return utils_test._dense_to_sparse(data)


def _batch_sparse_eye(batch_shape, num_vertices, dtype):
  """Generate a batch of identity matrices."""
  eye = np.eye(num_vertices, dtype=dtype)
  num_batch_dims = len(batch_shape)
  expand_shape = np.concatenate((np.ones((num_batch_dims), dtype=np.int32),
                                 (num_vertices, num_vertices)), axis=0)
  eye = np.reshape(eye, expand_shape)
  tile_shape = np.concatenate((batch_shape, (1, 1)), axis=0)
  return _dense_to_sparse(np.tile(eye, tile_shape))


class GraphPoolingTestPoolTests(test_case.TestCase):

  @parameterized.parameters(
      ("'sizes' must have an integer type.", np.float32, np.float32,
       np.float32),
      ("'data' must have a float type.", np.int32, np.float32, np.int32),
      ("'pool_map' and 'data' must have the same type.", np.float32, np.float64,
       np.int32)
  )
  def test_pool_exception_raised_types(
      self, err_msg, data_type, pool_map_type, sizes_type):
    """Tests the correct exceptions are raised for invalid types."""
    data = np.ones((2, 3, 3), dtype=data_type)
    pool_map = _dense_to_sparse(np.ones((2, 3, 3), dtype=pool_map_type))
    sizes = np.array(((1, 2), (2, 3)), dtype=sizes_type)

    with self.assertRaisesRegexp(TypeError, err_msg):
      gp.pool(data, pool_map, sizes)

  @parameterized.parameters(
      ('data must have a rank greater than 1', (3,), (3,), None),
      ('pool_map must have a rank of 2', (3, 3), (3,), None),
      ('sizes must have a rank of 3', (4, 5, 3, 2), (4, 5, 3, 3), (3, 2)),
  )
  def test_pool_exception_raised_shapes(
      self, err_msg, data_shape, pool_map_shape, sizes_shape):
    """Tests the correct exceptions are raised for invalid shapes."""
    data = np.ones(data_shape, dtype=np.float32)
    pool_map = _dense_to_sparse(np.ones(pool_map_shape, dtype=np.float32))
    if sizes_shape is not None:
      sizes = np.ones(sizes_shape, dtype=np.int32)
    else:
      sizes = None

    with self.assertRaisesRegexp(ValueError, err_msg):
      gp.pool(data, pool_map, sizes)

  def test_pool_exception_raised_algorithm(self):
    """Tests the correct exception is raised for an invalid algorithm."""
    data = np.ones(shape=(2, 2))
    pool_map = _dense_to_sparse(np.ones(shape=(2, 2)))

    with self.assertRaisesRegexp(
        ValueError, 'The pooling method must be "weighted" or "max"'):
      gp.pool(data, pool_map, sizes=None, algorithm='mean')

  @parameterized.parameters(
      ((2, 3), 4, 3, np.float32),
      ((1,), 6, 1, np.float32),
      ((4, 1, 3), 9, 7, np.float64),
      ((2, 8, 4, 6), 19, 11, np.float64),
  )
  def test_pool_identity(self,
                         batch_shape,
                         num_vertices,
                         num_features,
                         data_type):
    """Tests graph pooling with identity maps."""
    data_shape = np.concatenate((batch_shape, (num_vertices, num_features)))
    data = np.random.uniform(size=data_shape).astype(data_type)
    pool_map = _batch_sparse_eye(batch_shape, num_vertices, data_type)

    pooled_max = gp.pool(
        data, pool_map, sizes=None, algorithm='max', name=None)
    pooled_weighted = gp.pool(
        data, pool_map, sizes=None, algorithm='weighted', name=None)

    self.assertAllClose(pooled_max, data)
    self.assertAllClose(pooled_weighted, data)

  def test_pool_preset_padded(self):
    """Tests pooling with preset data and padding."""
    data = np.reshape(np.arange(12).astype(np.float32), (2, 3, 2))
    sizes = ((2, 3), (3, 3))
    pool_map = _dense_to_sparse(np.array(
        (((0.5, 0.5, 0.),
          (0., 0., 1.),
          (0., 0., 0.)),
         ((1., 0., 0.),
          (0., 1., 0.),
          (0., 0., 1.))), dtype=np.float32))

    pooled_max = gp.pool(data, pool_map, sizes, algorithm='max')
    pooled_weighted = gp.pool(data,
                              pool_map,
                              sizes,
                              algorithm='weighted')
    true_max = (((2., 3.), (4., 5.), (0., 0.)),
                ((6., 7.), (8., 9.), (10., 11.)))
    true_weighted = (((1., 2.), (4., 5.), (0., 0.)),
                     ((6., 7.), (8., 9.), (10., 11.)))

    self.assertAllClose(pooled_max, true_max)
    self.assertAllClose(pooled_weighted, true_weighted)

  def test_pool_preset(self):
    """Tests pooling with preset data."""
    pool_map = np.array(((0.5, 0.5, 0., 0.),
                         (0., 0., 0.5, 0.5)), dtype=np.float32)
    pool_map = _dense_to_sparse(pool_map)
    data = np.reshape(np.arange(8).astype(np.float32), (4, 2))
    max_true = data[(1, 3), :]
    max_weighted = (data[(0, 2), :] + max_true) * 0.5

    pooled_max = gp.pool(
        data, pool_map, sizes=None, algorithm='max', name=None)
    pooled_weighted = gp.pool(
        data, pool_map, sizes=None, algorithm='weighted', name=None)

    self.assertAllClose(pooled_max, max_true)
    self.assertAllClose(pooled_weighted, max_weighted)

  @parameterized.parameters((20, 10, 3), (2, 1, 1), (2, 5, 4), (2, 1, 3))
  def test_pool_random(
      self, num_input_vertices, num_output_vertices, num_features):
    """Tests pooling with random inputs."""
    pool_map = 0.001 + np.random.uniform(
        size=(num_output_vertices, num_input_vertices))
    data = np.random.uniform(size=(num_input_vertices, num_features))
    true_weighted = np.matmul(pool_map, data)
    true_max = np.tile(np.max(data, axis=0, keepdims=True),
                       (num_output_vertices, 1))
    pool_map = _dense_to_sparse(pool_map)

    with self.subTest(name='max'):
      pooled_max = gp.pool(data, pool_map, None, algorithm='max')
      self.assertAllClose(pooled_max, true_max)

    with self.subTest(name='weighted'):
      pooled_weighted = gp.pool(
          data, pool_map, None, algorithm='weighted')
      self.assertAllClose(pooled_weighted, true_weighted)

  def test_pool_jacobian(self):
    """Tests the jacobian is correct."""
    sizes = ((2, 4), (3, 5))
    data_init = np.random.uniform(size=(2, 5, 3))
    pool_map = np.random.uniform(size=(2, 3, 5))
    data_init[0, -1, :] = 0.
    pool_map[0, -1, :] = 0.
    pool_map = _dense_to_sparse(pool_map)
    data = tf.convert_to_tensor(value=data_init)

    with self.subTest(name='max'):
      pooled_max = gp.pool(data, pool_map, sizes, algorithm='max')
      self.assert_jacobian_is_correct(data, data_init, pooled_max)

    with self.subTest(name='weighted'):
      pooled_weighted = gp.pool(data,
                                pool_map,
                                sizes,
                                algorithm='weighted')
      self.assert_jacobian_is_correct(data, data_init, pooled_weighted)


if __name__ == '__main__':
  test_case.main()
