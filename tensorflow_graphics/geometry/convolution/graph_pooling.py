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
"""This module implements various graph pooling ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.geometry.convolution import utils
from tensorflow_graphics.util import export_api


def pool(data, pool_map, sizes, algorithm='max', name=None):
  #  pyformat: disable
  """Implements graph pooling.

  The features at each output vertex are computed by pooling over a subset of
  vertices in the input graph. This pooling window is specified by the input
  `pool_map`.

  The shorthands used below are
    `V1`: The number of vertices in the input data.
    `C`: The number of channels in the data.
    `V2`: The number of vertices in the pooled output data.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    data: A `float` tensor with shape `[A1, ..., An, V1, C]`.
    pool_map: A `SparseTensor` with the same type as `data` and with shape
      `[A1, ..., An, V2, V1]`. The features for an output vertex `v2` will be
      computed by pooling over the corresponding input vertices specified by
      the entries in `pool_map[A1, ..., An, v2, :]`.
    sizes: An `int` tensor of shape `[A1, ..., An, 2]` indicating the true
      input sizes in case of padding (`sizes=None` indicates no padding).
      `sizes[A1, ..., An, 0] <= V2` specifies the padding in the (pooled)
      output, and `sizes[A1, ..., An, 1] <= V1` specifies the padding in the
      input.
    algorithm: The pooling function, must be either 'max' or 'weighted'. Default
      is 'max'. For 'max' pooling, the output features are the maximum over the
      input vertices (in this case only the indices of the `SparseTensor`
      `pool_map` are used, the values are ignored). For 'weighted', the output
      features are a weighted sum of the input vertices, the weights specified
      by the values of `pool_map`.
    name: A name for this op. Defaults to 'graph_pooling_pool'.

  Returns:
    Tensor with shape `[A1, ..., An, V2, C]`.

  Raises:
    TypeError: if the input types are invalid.
    ValueError: if the input dimensions are invalid.
    ValueError: if `algorithm` is invalid.
  """
  #  pyformat: enable
  with tf.compat.v1.name_scope(
      name, 'graph_pooling_pool', [data, pool_map, sizes]):
    data = tf.convert_to_tensor(value=data)
    pool_map = tf.compat.v1.convert_to_tensor_or_sparse_tensor(value=pool_map)
    if sizes is not None:
      sizes = tf.convert_to_tensor(value=sizes)
    utils.check_valid_graph_pooling_input(data, pool_map, sizes)

    if sizes is not None:
      sizes_output, sizes_input = tf.split(sizes, 2, axis=-1)
      sizes_output = tf.squeeze(sizes_output, axis=-1)
      sizes_input = tf.squeeze(sizes_input, axis=-1)
    else:
      sizes_output = None
      sizes_input = None

    batched = data.shape.ndims > 2
    if batched:
      x_flat, _ = utils.flatten_batch_to_2d(data, sizes_input)
      pool_map_block_diagonal = utils.convert_to_block_diag_2d(pool_map, sizes)
    else:
      x_flat = data
      pool_map_block_diagonal = pool_map

    if algorithm == 'weighted':
      pooled = tf.sparse.sparse_dense_matmul(
          pool_map_block_diagonal, x_flat)
    elif algorithm == 'max':
      pool_groups = tf.gather(x_flat, pool_map_block_diagonal.indices[:, 1])
      ragged_groups = tf.RaggedTensor.from_value_rowids(
          values=pool_groups,
          value_rowids=pool_map_block_diagonal.indices[:, 0])
      pooled = tf.reduce_max(input_tensor=ragged_groups, axis=1)
    else:
      raise ValueError('The pooling method must be "weighted" or "max"')

    if batched:
      if sizes_output is not None:
        pooled = utils.unflatten_2d_to_batch(pooled, sizes_output)
      else:
        output_shape = tf.concat((tf.shape(input=pool_map)[:-1], (-1,)), axis=0)
        pooled = tf.reshape(pooled, output_shape)

    return pooled

# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
