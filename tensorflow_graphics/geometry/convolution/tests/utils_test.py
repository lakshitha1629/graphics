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
"""Tests for convolution utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import scipy as sp
import tensorflow as tf

from tensorflow_graphics.geometry.convolution import utils
from tensorflow_graphics.util import test_case


def _dense_to_sparse(data):
  """Converts a numpy array to a tf.SparseTensor."""
  indices = np.where(data)
  return tf.SparseTensor(
      np.stack(indices, axis=-1), data[indices], dense_shape=data.shape)


class UtilsCheckValidGraphConvolutionInputTests(test_case.TestCase):

  @parameterized.parameters(
      ("'sizes' must have an integer type.", np.float32, np.float32,
       np.float32),
      ("'data' must have a float type.", np.int32, np.float32, np.int32),
      ("'neighbors' and 'data' must have the same type.", np.float32,
       np.float64, np.int32),
  )
  def test_check_valid_graph_convolution_input_exception_raised_types(
      self, err_msg, data_type, neighbors_type, sizes_type):
    """Check the type errors for invalid input types."""
    data = tf.convert_to_tensor(
        value=np.random.uniform(size=(2, 2, 2)).astype(data_type))
    neighbors = _dense_to_sparse(np.ones(shape=(2, 2, 2), dtype=neighbors_type))
    sizes = tf.convert_to_tensor(value=np.array((2, 2), dtype=sizes_type))

    with self.assertRaisesRegexp(TypeError, err_msg):
      utils.check_valid_graph_convolution_input(data, neighbors, sizes)

  @parameterized.parameters(
      (np.float32, np.float32, np.int32),
      (np.float64, np.float64, np.int32),
      (np.float32, np.float32, np.int64),
      (np.float64, np.float64, np.int64),
  )
  def test_check_valid_graph_convolution_input_exception_not_raised_types(
      self, data_type, neighbors_type, sizes_type):
    """Check there are no exceptions for valid input types."""
    data = tf.convert_to_tensor(
        value=np.random.uniform(size=(2, 2, 2)).astype(data_type))
    neighbors = _dense_to_sparse(np.ones(shape=(2, 2, 2), dtype=neighbors_type))
    sizes = tf.convert_to_tensor(value=np.array((2, 2), dtype=sizes_type))

    self.assert_exception_is_not_raised(
        utils.check_valid_graph_convolution_input,
        shapes=[],
        data=data,
        neighbors=neighbors,
        sizes=sizes)

  @parameterized.parameters(
      ((2, 3), (2, 2), None),
      ((1, 2, 3), (1, 2, 2), None),
      ((2, 2, 3), (2, 2, 2), None),
      ((1, 2, 3), (1, 2, 2), (1,)),
      ((2, 2, 3), (2, 2, 2), (2,)),
      ((1, 2, 2, 3), (1, 2, 2, 2), (1, 2)),
      ((2, 2, 2, 3), (2, 2, 2, 2), (2, 2)),
  )
  def test_check_valid_graph_convolution_input_exception_not_raised_shapes(
      self, *shapes):
    """Check that valid input shapes do not trigger any exceptions."""
    data = tf.convert_to_tensor(
        value=np.random.uniform(size=shapes[0]).astype(np.float32))
    neighbors = _dense_to_sparse(np.ones(shape=shapes[1], dtype=np.float32))
    if shapes[2] is not None:
      sizes = tf.convert_to_tensor(
          value=np.ones(shape=shapes[2], dtype=np.int32))
    else:
      sizes = None

    self.assert_exception_is_not_raised(
        utils.check_valid_graph_convolution_input,
        shapes=[],
        data=data,
        neighbors=neighbors,
        sizes=sizes)

  @parameterized.parameters(
      ("must have a rank of 2", (5, 2), (1, 5, 2), None),
      ("must have a rank greater than 1", (5,), (5, 5), None),
      ("must have a rank of 2", (5, 2), (5,), None),
      ("must have a rank of 3", (5, 5, 2), (5, 5), None),
      ("must have the same number of dimensions in axes", (3, 2), (3, 2), None),
      ("must have a rank of 1", (5, 5, 2), (5, 5, 5), (5, 5)),
      ("Not all batch dimensions are identical.", (1, 5, 2), (1, 5, 5), (2,)),
  )
  def test_check_valid_graph_convolution_input_exception_raised_shapes(
      self, error_msg, *shapes):
    """Check that invalid input shapes trigger the right exceptions."""
    data = tf.convert_to_tensor(
        value=np.random.uniform(size=shapes[0]).astype(np.float32))
    neighbors = _dense_to_sparse(np.ones(shape=shapes[1], dtype=np.float32))
    if shapes[2] is not None:
      sizes = tf.convert_to_tensor(
          value=np.ones(shape=shapes[2], dtype=np.int32))
    else:
      sizes = None

    self.assert_exception_is_raised(
        utils.check_valid_graph_convolution_input,
        error_msg,
        shapes=[],
        data=data,
        neighbors=neighbors,
        sizes=sizes)


class UtilsFlattenBatchTo2dTests(test_case.TestCase):

  @parameterized.parameters(((5, 3),), ((3,),))
  def test_input_rank_exception_raised(self, *shapes):
    """Check that invalid input data rank triggers the right exception."""
    self.assert_exception_is_raised(utils.flatten_batch_to_2d,
                                    "must have a rank greater than 2", shapes)

  def test_flatten_batch_to_2d_exception_raised_types(self):
    """Check the exception when input is not a SparseTensor."""
    with self.assertRaisesRegexp(TypeError,
                                 "'sizes' must have an integer type."):
      utils.flatten_batch_to_2d(np.ones((3, 4, 3)), np.ones((3,)))

  @parameterized.parameters(
      ("must have a rank of 1", (3, 4, 3), (3, 4)),
      ("must have a rank of 1", (3, 4, 5), (3, 4, 5)),
  )
  def test_flatten_batch_to_2d_exception_raised(self, error_msg, *shapes):
    """Check the exception when the shape of 'sizes' is invalid."""
    self.assert_exception_is_raised(
        utils.flatten_batch_to_2d,
        error_msg,
        shapes,
        dtypes=(tf.float32, tf.int32))

  def test_flatten_batch_to_2d_random(self):
    """Test flattening with random inputs."""
    ndims_batch = np.random.randint(low=1, high=5)
    batch_dims = np.random.randint(low=1, high=10, size=ndims_batch)
    data_dims = np.random.randint(low=1, high=20, size=2)
    dims = np.concatenate([batch_dims, data_dims], axis=0)
    data = np.random.uniform(size=dims)

    with self.subTest(name="random_padding"):
      sizes = np.random.randint(low=0, high=data_dims[0], size=batch_dims)
      y, unflatten = utils.flatten_batch_to_2d(data, sizes)
      data_unflattened = unflatten(y)

      self.assertAllEqual(tf.shape(input=y), [np.sum(sizes), data_dims[1]])
      self.assertAllEqual(
          tf.shape(input=data_unflattened), tf.shape(input=data))

    with self.subTest(name="no_padding_with_sizes"):
      sizes = data_dims[0] * np.ones_like(sizes, dtype=np.int32)
      y, unflatten = utils.flatten_batch_to_2d(data, sizes)

      self.assertAllEqual(tf.shape(input=y), [np.sum(sizes), data_dims[1]])
      self.assertAllEqual(data, unflatten(y))

    with self.subTest(name="no_padding_with_sizes_none"):
      y, unflatten = utils.flatten_batch_to_2d(data, sizes=None)

      self.assertAllEqual(tf.shape(input=y), [np.sum(sizes), data_dims[1]])
      self.assertAllEqual(data, unflatten(y))

  def test_flatten_batch_to_2d_zero_sizes(self):
    """Test flattening with zero sizes."""
    data = np.ones(shape=(10, 5, 3, 2))
    sizes = np.zeros(shape=(10, 5), dtype=np.int32)

    y, unflatten = utils.flatten_batch_to_2d(data, sizes)

    self.assertAllEqual([0, 2], tf.shape(input=y))
    self.assertAllEqual(np.zeros_like(data), unflatten(y))

  def test_flatten_batch_to_2d_unflatten_different_feature_dims(self):
    """Test when inputs to flattening/unflattening use different channels."""
    data_in = np.random.uniform(size=(3, 1, 7, 5, 4))
    data_out = np.concatenate([data_in, data_in], axis=-1)

    y, unflatten = utils.flatten_batch_to_2d(data_in)

    self.assertAllEqual(unflatten(tf.concat([y, y], axis=-1)), data_out)

  def test_flatten_batch_to_2d_jacobian_random(self):
    """Test the jacobian is correct for random inputs."""
    data_init = np.random.uniform(size=(3, 2, 7, 5, 4))
    sizes = np.random.randint(low=1, high=5, size=(3, 2, 7))
    data = tf.convert_to_tensor(value=data_init)

    y, unflatten = utils.flatten_batch_to_2d(data, sizes=sizes)
    # assert_jacobian_is_correct() requires a fully defined static shape.
    y.set_shape([np.sum(sizes), 4])
    flat_init = np.random.uniform(size=(np.sum(sizes), 10))
    flat = tf.convert_to_tensor(value=flat_init)
    y_unflattened = unflatten(flat)

    self.assert_jacobian_is_correct(data, data_init, y)
    self.assert_jacobian_is_correct(flat, flat_init, y_unflattened)

  @parameterized.parameters((np.int32), (np.float32), (np.uint16))
  def test_flatten_batch_to_2d_unflatten_types(self, dtype):
    """Test unflattening with int and float types."""
    data = np.ones(shape=(2, 2, 3, 2), dtype=dtype)
    sizes = ((3, 2), (1, 3))
    desired_unflattened = data
    desired_unflattened[0, 1, 2, :] = 0
    desired_unflattened[1, 0, 1:, :] = 0

    flat, unflatten = utils.flatten_batch_to_2d(data, sizes=sizes)
    data_unflattened = unflatten(flat)

    self.assertEqual(data.dtype, data_unflattened.dtype.as_numpy_dtype)
    self.assertAllEqual(data_unflattened, desired_unflattened)


class UtilsConvertToBlockDiag2dTests(test_case.TestCase):

  def _dense_to_sparse(self, x):
    """Convert a numpy array to a tf.SparseTensor."""
    indices = np.where(x)
    return tf.SparseTensor(
        np.stack(indices, axis=-1), x[indices], dense_shape=x.shape)

  def _validate_sizes(self, block_diag_tensor, sizes):
    """Assert all elements outside the blocks are zero."""
    data = [np.ones(shape=s) for s in sizes]
    mask = 1.0 - sp.linalg.block_diag(*data)

    self.assertAllEqual(
        tf.sparse.to_dense(block_diag_tensor) * mask, np.zeros_like(mask))

  def test_convert_to_block_diag_2d_exception_raised_types(self):
    """Check the exception when input is not a SparseTensor."""
    with self.assertRaisesRegexp(TypeError, "'data' must be a 'SparseTensor'."):
      utils.convert_to_block_diag_2d(np.zeros(shape=(3, 3, 3)))

    with self.assertRaisesRegexp(TypeError,
                                 "'sizes' must have an integer type."):
      utils.convert_to_block_diag_2d(
          self._dense_to_sparse(np.ones(shape=(3, 3, 3))),
          np.ones(shape=(3, 2)),
      )

  def test_convert_to_block_diag_2d_exception_raised_ranks(self):
    """Check the exception when input data rank is invalid."""
    with self.assertRaisesRegexp(ValueError, "must have a rank greater than 2"):
      utils.convert_to_block_diag_2d(
          self._dense_to_sparse(np.ones(shape=(3, 3))))

    with self.assertRaisesRegexp(ValueError, "must have a rank greater than 2"):
      utils.convert_to_block_diag_2d(self._dense_to_sparse(np.ones(shape=(3,))))

  def test_convert_to_block_diag_2d_exception_raised_sizes(self):
    """Check the expetion when the shape of sizes is invalid."""
    with self.assertRaisesRegexp(ValueError, "must have a rank of 2"):
      utils.convert_to_block_diag_2d(
          self._dense_to_sparse(np.ones(shape=(3, 3, 3))),
          np.ones(shape=(3,), dtype=np.int32))

    with self.assertRaisesRegexp(ValueError, "must have a rank of 3"):
      utils.convert_to_block_diag_2d(
          self._dense_to_sparse(np.ones(shape=(4, 3, 3, 3))),
          np.ones(shape=(4, 3), dtype=np.int32))

    with self.assertRaisesRegexp(ValueError,
                                 "must have exactly 2 dimensions in axis -1"):
      utils.convert_to_block_diag_2d(
          self._dense_to_sparse(np.ones(shape=(3, 3, 3))),
          np.ones(shape=(3, 1), dtype=np.int32))

  def test_convert_to_block_diag_2d_random(self):
    """Test block diagonalization with random inputs."""
    sizes = np.random.randint(low=2, high=6, size=(3, 2))
    data = [np.random.uniform(size=s) for s in sizes]
    batch_data_padded = np.zeros(
        shape=np.concatenate(([len(sizes)], np.max(sizes, axis=0)), axis=0))
    for i, s in enumerate(sizes):
      batch_data_padded[i, :s[0], :s[1]] = data[i]

    batch_data_padded_sparse = self._dense_to_sparse(batch_data_padded)
    block_diag_data = sp.linalg.block_diag(*data)
    block_diag_sparse = utils.convert_to_block_diag_2d(
        batch_data_padded_sparse, sizes=sizes)

    self.assertAllEqual(tf.sparse.to_dense(block_diag_sparse), block_diag_data)

  def test_convert_to_block_diag_2d_no_padding(self):
    """Test block diagonalization without any padding."""
    batch_data = np.random.uniform(size=(3, 4, 5, 4))
    block_diag_data = sp.linalg.block_diag(
        *[x for x in np.reshape(batch_data, (-1, 5, 4))])

    batch_data_sparse = self._dense_to_sparse(batch_data)
    block_diag_sparse = utils.convert_to_block_diag_2d(batch_data_sparse)

    self.assertAllEqual(tf.sparse.to_dense(block_diag_sparse), block_diag_data)

  def test_convert_to_block_diag_2d_validate_indices(self):
    """Test block diagonalization when we filter out out-of-bounds indices."""
    sizes = ((2, 3), (2, 3), (2, 3))

    batch = self._dense_to_sparse(np.random.uniform(size=(3, 4, 3)))
    block_diag = utils.convert_to_block_diag_2d(batch, sizes, True)

    self._validate_sizes(block_diag, sizes)

  def test_convert_to_block_diag_2d_large_sizes(self):
    """Test when the desired blocks are larger than the data shapes."""
    sizes = ((5, 5), (6, 6), (7, 7))

    batch = self._dense_to_sparse(np.random.uniform(size=(3, 4, 3)))
    block_diag = utils.convert_to_block_diag_2d(batch, sizes)

    self._validate_sizes(block_diag, sizes)

  def test_convert_to_block_diag_2d_batch_shapes(self):
    """Test with different batch shapes."""
    sizes_one_batch_dim = np.concatenate(
        [np.random.randint(low=1, high=h, size=(6 * 3 * 4, 1)) for h in (5, 7)],
        axis=-1)
    data = [np.random.uniform(size=s) for s in sizes_one_batch_dim]
    data_one_batch_dim_padded = np.zeros(shape=(6 * 3 * 4, 5, 7))
    for i, s in enumerate(sizes_one_batch_dim):
      data_one_batch_dim_padded[i, :s[0], :s[1]] = data[i]
    data_many_batch_dim_padded = np.reshape(data_one_batch_dim_padded,
                                            (6, 3, 4, 5, 7))
    sizes_many_batch_dim = np.reshape(sizes_one_batch_dim, (6, 3, 4, -1))

    data_one_sparse = self._dense_to_sparse(data_one_batch_dim_padded)
    data_many_sparse = self._dense_to_sparse(data_many_batch_dim_padded)
    one_batch_dim = utils.convert_to_block_diag_2d(data_one_sparse,
                                                   sizes_one_batch_dim)
    many_batch_dim = utils.convert_to_block_diag_2d(data_many_sparse,
                                                    sizes_many_batch_dim)

    self.assertAllEqual(
        tf.sparse.to_dense(one_batch_dim), tf.sparse.to_dense(many_batch_dim))
    self._validate_sizes(one_batch_dim, sizes_one_batch_dim)

  def test_convert_to_block_diag_2d_jacobian_random(self):
    """Test the jacobian is correct with random inputs."""
    sizes = np.random.randint(low=2, high=6, size=(3, 2))
    data = [np.random.uniform(size=s) for s in sizes]
    batch_data_padded = np.zeros(
        shape=np.concatenate([[len(sizes)], np.max(sizes, axis=0)], axis=0))
    for i, s in enumerate(sizes):
      batch_data_padded[i, :s[0], :s[1]] = data[i]
    sparse_ind = np.where(batch_data_padded)
    sparse_val_init = batch_data_padded[sparse_ind]
    sparse_val = tf.convert_to_tensor(value=sparse_val_init)
    sparse = tf.SparseTensor(
        np.stack(sparse_ind, axis=-1), sparse_val, batch_data_padded.shape)

    y = utils.convert_to_block_diag_2d(sparse, sizes)

    self.assert_jacobian_is_correct(sparse_val, sparse_val_init, y.values)


class UtilsPartitionSums2dTests(test_case.TestCase):

  def _numpy_ground_truth(self, data, group_ids, row_weights):
    """Ground truth row sum in numpy."""
    out_weighted = np.zeros([np.max(group_ids) + 1, data.shape[1]])
    out_uniform = np.zeros_like(out_weighted)
    for i, val in enumerate(group_ids):
      out_uniform[val] += data[i]
      out_weighted[val] += row_weights[i] * data[i]
    return out_weighted, out_uniform

  @parameterized.parameters(
      ("must have a rank of 2", ((3, 4, 5), (3,)), (tf.float32, tf.int32)),
      ("must have a rank of 2", ((3,), (3,)), (tf.float32, tf.int32)),
      ("must have a rank of 1", ((3, 3), (3, 1)), (tf.float32, tf.int32)),
      ("must have a rank of 1", ((3, 3), (3,), (3, 1)),
       (tf.float32, tf.int32, tf.float32)),
  )
  def test_partition_sums_2d_input_shapes(self, error_msg, shapes, dtypes):
    self.assert_exception_is_raised(utils.partition_sums_2d, error_msg, shapes,
                                    dtypes)

  @parameterized.parameters(
      ("'data' and 'row_weights' must have the same type.",
       (tf.float32, tf.int32, tf.float64)),
      ("'group_ids' must be an integer tensor.",
       (tf.float32, tf.float32, tf.float32)),
  )
  def test_partition_sums_2d_exception_raised_types(self, error_msg, dtypes):
    """Check the exceptions with invalid input types."""
    x1 = tf.zeros(shape=(3, 3), dtype=dtypes[0])
    x2 = tf.zeros(shape=(3), dtype=dtypes[1])
    x3 = tf.zeros(shape=(3), dtype=dtypes[2])

    with self.assertRaisesRegexp(TypeError, error_msg):
      utils.partition_sums_2d(x1, x2, x3)

  def test_partition_sums_2d_single_output_dim(self):
    """Test when there is a single output dimension."""
    data = np.ones([10, 10])
    group_ids = np.zeros([10], dtype=np.int32)
    gt = np.sum(data, axis=0, keepdims=True)

    self.assertAllEqual(gt, utils.partition_sums_2d(data, group_ids))

  def test_partition_sums_2d_random(self):
    """Test with random inputs."""
    data_shape = np.random.randint(low=50, high=100, size=2)
    data = np.random.uniform(size=data_shape)
    group_ids = np.random.randint(
        low=0, high=np.random.randint(low=50, high=150), size=data_shape[0])
    row_weights = np.random.uniform(size=group_ids.shape)

    gt_weighted, gt_uniform = self._numpy_ground_truth(data, group_ids,
                                                       row_weights)
    weighted_sum = utils.partition_sums_2d(
        data, group_ids, row_weights=row_weights)
    uniform_sum = utils.partition_sums_2d(data, group_ids, row_weights=None)

    self.assertAllClose(weighted_sum, gt_weighted)
    self.assertAllClose(uniform_sum, gt_uniform)

  def test_partition_sums_2d_jacobian_random(self):
    """Test the jacobian with random inputs."""
    data_shape = np.random.randint(low=5, high=10, size=2)
    data_init = np.random.uniform(size=data_shape)
    data = tf.convert_to_tensor(value=data_init)
    group_ids = np.random.randint(
        low=0, high=np.random.randint(low=50, high=150), size=data_shape[0])
    row_weights_init = np.random.uniform(size=group_ids.shape)
    row_weights = tf.convert_to_tensor(value=row_weights_init)

    y = utils.partition_sums_2d(data, group_ids, row_weights)

    self.assert_jacobian_is_correct(data, data_init, y)
    self.assert_jacobian_is_correct(row_weights, row_weights_init, y)


if __name__ == "__main__":
  test_case.main()
