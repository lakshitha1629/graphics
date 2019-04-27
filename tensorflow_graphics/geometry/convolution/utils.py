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
"""This module implements various sparse data utilities for graphs and meshes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import export_api


def check_valid_graph_convolution_input(data, neighbors, sizes):
  """Checks that the inputs are valid for graph convolution ops.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    data: A `float` tensor with shape `[A1, ..., An, V1, V2]`.
    neighbors: A SparseTensor with the same type as `data` and with
      shape `[A1, ..., An, V1, V1]`.
    sizes: An `int` tensor of shape `[A1, ..., An]`. Optional, can be `None`.

  Raises:
    TypeError: if the input types are invalid.
    ValueError: if the input dimensions are invalid.
  """
  # We expect the inputs to be Tensors so we skip convert_to_tensor().
  if not data.dtype.is_floating:
    raise TypeError("'data' must have a float type.")
  if neighbors.dtype != data.dtype:
    raise TypeError("'neighbors' and 'data' must have the same type.")
  data_ndims = data.shape.ndims
  if data_ndims != neighbors.shape.ndims:
    raise ValueError("'data' and 'neighbors' must have the same rank.")
  if data_ndims < 2:
    raise ValueError("'data' and 'neighbors' must have rank >= 2.")
  if sizes is not None:
    if not sizes.dtype.is_integer:
      raise TypeError("'sizes' must have an integer type.")
    if data_ndims != sizes.shape.ndims + 2:
      raise ValueError(
          "'sizes' shape and the batch shape of 'data' must be equal.")


def flatten_batch_to_2d(data, sizes=None, name=None):
  """Reshape a batch of 2d Tensors by flattening across the batch dimensions.

  Note:
    In the following, A1 to An are optional batch dimensions.

  A tensor with shape `[A1, ..., An, D1, D2]` will be reshaped to one
  with shape `[A1*...*An*D1, D2]`. This function also returns an inverse
  function that returns any tensor with shape `[A1*...*An*D1, D3]` to one
  with shape `[A1, ..., An, D1, D3]`.

  Padded inputs in dim D1 are allowed. `sizes` determines the first elements
  from D1 to select from each batch dimension.

  Examples:

      data = [[[1., 2.], [3., 4.]],
              [[5., 6.], [7., 8.]],
              [[9., 10.], [11., 12.]]]
      sizes = None
      output = [[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.], [11., 12.]]
      unflatten(output) = data

      data = [[[1., 2.], [0., 0.]],
              [[5., 6.], [7., 8.]],
              [[9., 10.], [0., 0.]]]
      sizes = [1, 2, 1]
      output = [[1., 2.], [5., 6.], [7., 8.], [9., 10.]]
      unflatten(output) = data


  Args:
    data: A tensor with shape `[A1, ..., An, D1, D2]`.
    sizes: An `int` tensor with shape `[A1, ..., An]`. Can be `None`.
      `sizes[i] <= D1`.
    name: A name for this op. Defaults to `utils_flatten_batch_to_2d`.

  Returns:
    A tensor with shape `[A1*...*An*D1, D2]` if `sizes==None`, otherwise a
      tensor  with shape `[sum(sizes), D2]`.
    A function that reshapes a tensor with shape `[A1*...*An*D1, D3]` to a
      tensor with shape `[A1, ..., An, D1, D3]` if `sizes==None`, otherwise it
      reshapes a tensor with shape `[sum(sizes), D3]` to one with shape
      `[A1, ..., An, ..., D1, D3]`.

  Raises:
    ValueError: if the input tensor dimensions are invalid.
  """
  with tf.compat.v1.name_scope(name, "utils_flatten_batch_to_2d",
                               [data, sizes]):
    data = tf.convert_to_tensor(value=data)
    data_ndims = data.shape.ndims
    if data_ndims < 3:
      raise ValueError("'data' must have rank > 2.")
    if sizes is not None:
      sizes = tf.convert_to_tensor(value=sizes)
      if sizes.shape.ndims != data_ndims - 2:
        raise ValueError(
            "'sizes' shape must match the batch shape of 'data'.")

    data_shape = tf.shape(input=data)
    if sizes is None:
      flat = tf.reshape(data, [-1, data_shape[-1]])
      def unflatten(flat):
        """Invert flatten_batch_to_2d."""
        flat = tf.convert_to_tensor(value=flat)
        output_shape = tf.concat(
            [data_shape[:-1], tf.shape(input=flat)[-1:]], axis=0)
        return tf.reshape(flat, output_shape)
    else:
      # Create a mask for the desired rows in `data` to select for flattening:
      # `mask` has shape `[A1, ..., An, D1]` and
      # `mask[a1, ..., an, :] = [True, ..., True, False, ..., False]` where
      # the number of True elements is `sizes[a1, ..., an]`.
      mask = tf.sequence_mask(sizes, data_shape[-2])
      mask_indices = tf.cast(tf.where(mask), tf.int32)
      flat = tf.gather_nd(params=data, indices=mask_indices)
      def unflatten(flat):
        """Invert flatten_batch_to_2d."""
        flat = tf.convert_to_tensor(value=flat)
        output_shape = tf.concat(
            [data_shape[:-1], tf.shape(input=flat)[-1:]], axis=0)
        return tf.scatter_nd(
            indices=mask_indices, updates=flat, shape=output_shape)

    return flat, unflatten


def convert_to_block_diag_2d(
    data, sizes=None, validate_indices=False, name=None):
  """Convert a batch of 2d SparseTensors to a 2d block diagonal SparseTensor.

  Note:
    In the following, A1 to An are optional batch dimensions.

  A SparseTensor with dense shape `[A1, ..., An, D1, D2]` will be reshaped
  to one with shape `[A1*...*An*D1, A1*...*An*D2]`.

  Padded inputs in dims D1 and D2 are allowed. `sizes` indicates the un-padded
  shape for each inner `[D1, D2]` matrix. The additional (padded) rows and
  columns will be omitted in the block diagonal output.

  If padded (`sizes != None`), the input should not contain any sparse indices
  outside the bounds indicated by `sizes`. Setting `validate_indices=True` will
  explicitly filter any invalid sparse indices before block diagonalization.

  Args:
    data: A SparseTensor with dense shape `[A1, ..., An, D1, D2]`.
    sizes: A tensor with shape `[A1, ..., An, 2]`. Can be `None` (indicates
      no padding). If not `None`, `sizes` indicates the true sizes (before
      padding) of the inner dimensions of `data`.
    validate_indices: A boolean. Ignored if `sizes==None`. If True,
      out-of-bounds indices in `data` are explicitly ignored, otherwise
      out-of-bounds indices will cause undefined behaviour.
    name: A name for this op. Defaults to `utils_convert_to_block_diag_2d`.

  Returns:
    A 2d block-diagonal SparseTensor.

  Raises:
    TypeError: if the input types are invalid.
    ValueError: if the input dimensions are invalid.
  """
  with tf.compat.v1.name_scope(
      name, "utils_convert_to_block_diag_2d",
      [data, sizes, validate_indices]):
    data = tf.compat.v1.convert_to_tensor_or_sparse_tensor(value=data)
    if not isinstance(data, tf.SparseTensor):
      raise TypeError("'data' must be a 'SparseTensor'.")
    data_ndims = data.shape.ndims
    if data_ndims < 3:
      raise ValueError("'data' must have rank > 2.")
    if sizes is not None:
      sizes = tf.convert_to_tensor(value=sizes)
      if sizes.shape.ndims != data_ndims - 1:
        raise ValueError("'sizes' has an unexpected shape.")
      if sizes.shape[-1] != 2:
        raise ValueError("innermost dimension of 'sizes' must be 2.")

    data_shape = tf.shape(input=data)
    data = tf.sparse.reshape(
        data, [-1, data_shape[-2], data_shape[-1]])
    indices = data.indices
    if sizes is not None:
      sizes = tf.cast(tf.reshape(sizes, [-1, 2]), tf.int64)
      if validate_indices:
        in_bounds = ~tf.reduce_any(
            input_tensor=indices[:, 1:] >= tf.gather(
                sizes, indices[:, 0]), axis=-1)
        indices = tf.boolean_mask(tensor=indices, mask=in_bounds)
        values = tf.boolean_mask(tensor=data.values, mask=in_bounds)
      else:
        values = data.values
      cumsum = tf.cumsum(sizes, axis=0, exclusive=True)
      index_shift = tf.gather(cumsum, indices[:, 0])
      indices = indices[:, 1:] + index_shift
      block_diag = tf.SparseTensor(
          indices, values, tf.reduce_sum(input_tensor=sizes, axis=0))
    else:
      data_shape = tf.shape(input=data, out_type=tf.int64)
      index_shift = tf.expand_dims(indices[:, 0], -1) * data_shape[1:]
      indices = indices[:, 1:] + index_shift
      block_diag = tf.SparseTensor(
          indices, data.values, data_shape[0] * data_shape[1:])
    return block_diag


def partition_sums_2d(data, group_ids, row_weights=None, name=None):
  """Sum over subsets of rows in a 2-D tensor.

  Args:
    data: 2-D tensor with shape `[D1, D2]`.
    group_ids: 1-D `int` tensor with shape `[D1]`.
    row_weights: 1-D tensor with shape `[D1]`. Can be `None`.
    name: A name for this op. Defaults to `utils_partition_sums_2d`.

  Returns:
    A 2-D tensor with shape `[max(group_ids) + 1, D2]` where
      `output[i, :] = sum(data[j, :] * weight[j] * 1(group_ids[j] == i)),
      1(.) is the indicator function.

  Raises:
    ValueError: if the inputs have invalid dimensions or types.
  """
  with tf.compat.v1.name_scope(
      name, "utils_partition_sums_2d", [data, group_ids, row_weights]):
    data = tf.convert_to_tensor(value=data)
    group_ids = tf.convert_to_tensor(value=group_ids)
    if not group_ids.dtype.is_integer:
      raise TypeError("'group_ids' must be an integer tensor.")
    elif group_ids.dtype != tf.int64:
      group_ids = tf.cast(group_ids, dtype=tf.int64)
    if row_weights is None:
      row_weights = tf.ones_like(group_ids, dtype=data.dtype)
    else:
      row_weights = tf.convert_to_tensor(value=row_weights)
    if row_weights.dtype != data.dtype:
      raise TypeError("'data' and 'row_weights' must have the same type.")
    if len(data.shape) != 2:
      raise ValueError("'data' must be a 2-D tensor.")
    if len(group_ids.shape) != 1:
      raise ValueError("'group_ids' must be a 1-D tensor.")
    if len(row_weights.shape) != 1:
      raise ValueError("'row_weights' must be a 1-D tensor.")
    num_rows = tf.size(input=group_ids, out_type=tf.int64)
    sparse_indices = tf.stack([group_ids, tf.range(num_rows)], axis=1)
    shape = [tf.reduce_max(input_tensor=group_ids) + 1, num_rows]
    sparse = tf.SparseTensor(sparse_indices, row_weights, dense_shape=shape)
    return tf.sparse.sparse_dense_matmul(sparse, data)

# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
