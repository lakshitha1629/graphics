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
"""This module implements various graph convolutions in TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.geometry.representation.mesh.convolution import utils
from tensorflow_graphics.util import export_api


def feature_steered_convolution(
    data, neighbors, sizes, variable_u, variable_v, variable_c, variable_w,
    variable_b, name=None):
  """Implements the Feature Steered graph convolution.

  FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis
  Nitika Verma, Edmond Boyer, Jakob Verbeek
  CVPR 2018
  https://arxiv.org/abs/1706.05206

  The shorthands used below are
    `V`: The number of vertices.
    `C`: The number of channels in the input data.
    `D`: The number of channels in the output after convolution.
    `W`: The number of weight matrices used in the convolution.
    The input variables (`variable_u`, `variable_v`, `variable_c`,
      `variable_w`, `variable_b`) correspond to the variables with the same
      names in the paper cited above.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    data: A float tensor with shape `[A1, ..., An, V, C]`.
    neighbors: A SparseTensor with the same type as `data` and with shape
      `[A1, ..., An, V, V]` representing vertex neighborhoods. The neighborhood
      of a vertex defines the support region for convolution. For a mesh, a
      common choice for the neighborhood of vertex i would be the vertices in
      the K-ring of i (including i itself).
      Each vertex must have at least one neighbor. For a faithful implementation
      of the FeaStNet convolution, neighbors should be a row-normalized weight
      matrix corresponding to the graph adjacency matrix with self-edges:
      `neighbors[A1, ..., An, i, j] > 0` if vertex j is a neighbor of vertex i,
      and `neighbors[A1, ..., An, i, i] > 0` for all i,
      and `sum(neighbors, axis=-1)[A1, ..., An, i] == 1.0 for all i`.
      These requirements are relaxed in this implementation.
    sizes: An int tensor of shape `[A1, ..., An]` indicating the true
      input sizes in case of padding (`sizes=None` indicates no padding).
      `sizes[A1, ..., An] <= V`. If `data` and `neighbors` are 2-D, `sizes` will
      be ignored. An example usage of `sizes`: consider an input consisting of
      three graphs G0, G1, and G2 with V0, V1, and V2 vertices respectively.
      The padded input would have the following shapes:
      `data.shape = [3, V, C]`, and `neighbors.shape = [3, V, V]`, where
      `V=max([V0, V1, V2])`. The true sizes of each graph will be specified by
      `sizes=[V0, V1, V2]` and `data[i, :Vi, :]` and `neighbors[i, :Vi, :Vi]`
      will be the vertex and neighborhood data of graph Gi. The SparseTensor
      `neighbors` should have no nonzero entries in the padded regions.
    variable_u: A 2-D tensor with shape `[C, W]`.
    variable_v: A 2-D tensor with shape `[C, W]`.
    variable_c: A 1-D tensor with shape `[W]`.
    variable_w: A 3-D tensor with shape `[W, C, D]`.
    variable_b: A 1-D tensor with shape `[D]`.
    name: A name for this op. Defaults to
      `graph_convolution_feature_steered_convolution`.

  Returns:
    Tensor with shape `[A1, ..., An, V, D]`.

  Raises:
    TypeError: if the input types are invalid.
    ValueError: if the input dimensions are invalid.
  """
  with tf.compat.v1.name_scope(
      name, 'graph_convolution_feature_steered_convolution',
      [data, neighbors, sizes, variable_u, variable_v, variable_c, variable_w,
       variable_b]):
    # Validate the input types and dimensions.
    data = tf.convert_to_tensor(value=data)
    neighbors = tf.compat.v1.convert_to_tensor_or_sparse_tensor(value=neighbors)
    if sizes is not None:
      sizes = tf.convert_to_tensor(value=sizes)
    utils.check_valid_graph_convolution_input(data, neighbors, sizes)

    # Flatten the batch dimensions and remove any vertex padding.
    data_ndims = data.shape.ndims
    if data_ndims > 2:
      if sizes is not None:
        sizes_square = tf.stack((sizes, sizes), axis=-1)
      else:
        sizes_square = None
      x_flat, unflatten = utils.flatten_batch_to_2d(data, sizes)
      adjacency = utils.convert_to_block_diag_2d(neighbors, sizes_square)
    else:
      x_flat = data
      adjacency = neighbors

    x_u = tf.matmul(x_flat, variable_u)
    x_v = tf.matmul(x_flat, variable_v)
    x_u_rep = tf.gather(x_u, adjacency.indices[:, 0])
    x_v_sep = tf.gather(x_v, adjacency.indices[:, 1])

    weights_q = tf.exp(x_u_rep + x_v_sep + tf.reshape(variable_c, (1, -1)))
    weights_q_sum = tf.reduce_sum(
        input_tensor=weights_q, axis=-1, keepdims=True)
    weights_q = weights_q / weights_q_sum

    y_sum = 0
    x_sep = tf.gather(x_flat, adjacency.indices[:, 1])
    weight_matrices = variable_c.shape.as_list()[0]
    for m in range(weight_matrices):
      q_m = weights_q[:, m]
      w_m = variable_w[m, :, :]
      # Compute `y_i_m = sum_{j in neighborhood(i)} q_m(x_i, x_j) * w_m * x_j`.
      y_m = tf.matmul(
          utils.partition_sums_2d(
              tf.expand_dims(q_m, 1) * x_sep,
              adjacency.indices[:, 0], adjacency.values), w_m)
      y_sum = y_sum + y_m

    y_out = y_sum + tf.reshape(variable_b, [1, -1])

    if data_ndims > 2:
      y_out = unflatten(y_out)

    return y_out


def feature_steered_convolution_layer(
    data, neighbors, sizes, translation_invariant=True,
    weight_matrices=8, output_channels=None,
    initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1),
    name=None, var_name=None):
  """Wraps the function `feature_steered_convolution` as a TensorFlow layer.

  The shorthands used below are
    `V`: The number of vertices.
    `C`: The number of channels in the input data.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    data: A float tensor with shape `[A1, ..., An, V, C]`.
    neighbors: A SparseTensor with the same type as `data` and with
      shape `[A1, ..., An, V, V]` representing vertex neighborhoods. The
      neighborhood of a vertex defines the support region for convolution. For
      a mesh, a common choice for the neighborhood of vertex i would be the
      vertices in the K-ring of i (including i itself).
      Each vertex must have at least one neighbor. For a faithful implementation
      of the FeaStNet paper, neighbors should be a row-normalized weight matrix
      corresponding to the graph adjacency matrix with self-edges:
      `neighbors[A1, ..., An, i, j] > 0` if vertex j is a neighbor of vertex i,
      and `neighbors[A1, ..., An, i, i] > 0` for all i,
      and `sum(neighbors, axis=-1)[A1, ..., An, i] == 1.0 for all i`.
      These requirements are relaxed in this implementation.
    sizes: An int tensor of shape `[A1, ..., An]` indicating the true
      input sizes in case of padding (`sizes=None` indicates no padding).
      `sizes[A1, ..., An] <= V`. If `data` and `neighbors` are 2-D, `sizes` will
      be ignored. An example usage of `sizes`: consider an input consisting of
      three graphs G0, G1, and G2 with V0, V1, and V2 vertices respectively.
      The padded input would have the following shapes:
      `data.shape = [3, V, C]`, and `neighbors.shape = [3, V, V]`, where
      `V=max([V0, V1, V2])`. The true sizes of each graph will be specified by
      `sizes=[V0, V1, V2]`, and `data[i, :Vi, :]` and `neighbors[i, :Vi, :Vi]`
      will be the vertex and neighborhood data of graph Gi. The SparseTensor
      `neighbors` should have no nonzero entries in the padded regions.
    translation_invariant: A boolean.
    weight_matrices: The number of weight matrices used in the convolution.
    output_channels: The number of channels in the output. Optional, if None
      then `output_channels = C`.
    initializer: An initializer for the trainable variables.
    name: A (name_scope) name for this op. Passed through to
      feature_steered_convolution().
    var_name: A (var_scope) name for the variables. Defaults to
      `graph_convolution_feature_steered_convolution_weights`.

  Returns:
    Tensor with shape `[A1, ..., An, V, output_channels]`.
  """
  with tf.compat.v1.variable_scope(
      var_name,
      default_name='graph_convolution_feature_steered_convolution_weights'):
    # Skips shape validation to avoid redundancy with
    # feature_steered_convolution().
    data = tf.convert_to_tensor(value=data)
    in_channels = data.get_shape().as_list()[-1]
    out_channels = in_channels if output_channels is None else output_channels

    variable_u = tf.compat.v1.get_variable(
        shape=(in_channels, weight_matrices), dtype=data.dtype,
        initializer=initializer, name='u')
    if translation_invariant:
      variable_v = -variable_u
    else:
      variable_v = tf.compat.v1.get_variable(
          shape=(in_channels, weight_matrices), dtype=data.dtype,
          initializer=initializer, name='v')
    variable_c = tf.compat.v1.get_variable(
        shape=(weight_matrices), dtype=data.dtype, initializer=initializer,
        name='c')
    variable_w = tf.compat.v1.get_variable(
        shape=(weight_matrices, in_channels, out_channels), dtype=data.dtype,
        initializer=initializer, name='w')
    variable_b = tf.compat.v1.get_variable(
        shape=(out_channels), dtype=data.dtype,
        initializer=initializer, name='b')

    return feature_steered_convolution(
        data=data, neighbors=neighbors, sizes=sizes, variable_u=variable_u,
        variable_v=variable_v, variable_c=variable_c, variable_w=variable_w,
        variable_b=variable_b, name=name)


def edge_convolution_template(
    data, neighbors, sizes, edge_function, edge_function_kwargs, name=None):
  r"""A template for edge convolutions.

  This function implements a general edge convolution for graphs of the form
  $$
  y_i = \sum_{j \in \mathcal{N}(i)} w_{ij} f(x_i, x_j)
  $$

  Where
  $$\mathcal{N}(i)$$ is the set of vertices in the neighborhood of vertex $$i$$,
  $$x_i \in \mathbb{R}^C$$  are the features at vertex $$i$$,
  $$w_{ij} \in \mathbb{R}$$ is the weight for the edge between vertex $$i$$ and
    vertex $$j$$, and finally
  $$f(x_i, x_j): \mathbb{R}^{C} \times \mathbb{R}^{C} \to \mathbb{R}^{D}$$ is a
    user-supplied function.

  The shorthands used below are
    `V`: The number of vertices.
    `C`: The number of channels in the input data.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    data: A float tensor with shape `[A1, ..., An, V, C]`.
    neighbors: A SparseTensor with the same type as `data` and with
      shape `[A1, ..., An, V, V]` representing vertex neighborhoods. The
      neighborhood of a vertex defines the support region for convolution.
      The value at `neighbors[?, ..., ?, i, j]` corresponds to the weight
      $$w_{ij}$$ above. Each vertex must have at least one neighbor.
    sizes: An int tensor of shape `[A1, ..., An, V, V]` indicating the true
      input sizes in case of padding (`sizes=None` indicates no padding).
      `sizes[A1, ..., An] <= V`. If `data` and `neighbors` are 2-D, `sizes` will
      be ignored. An example usage of `sizes`: consider an input consisting of
      three graphs G0, G1, and G2 with V0, V1, and V2 vertices respectively.
      The padded input would have the following shapes:
      `data.shape = [3, V, C]`, and `neighbors.shape = [3, V, V]`, where
      `V=max([V0, V1, V2])`. The true sizes of each graph will be specified by
      `sizes=[V0, V1, V2]` and `data[i, :Vi, :]` and `neighbors[i, :Vi, :Vi]`
      will be the vertex and neighborhood data of graph Gi. The SparseTensor
      `neighbors` should have no nonzero entries in the padded regions.
    edge_function: A callable that takes at least two arguments of vertex
      features and returns a tensor of vertex features.
      `Y = f(X1, X2, **kwargs)`, where `X1` and `X2` have shape `[V3, C]` and
      `Y` must have shape `[V3, D], D >= 1`.
    edge_function_kwargs: A dict containing any additional keyword arguments to
      be passed to `edge_function`.
    name: A name for this op. Defaults to
      `graph_convolution_edge_convolution_template`.

  Returns:
    Tensor with shape `[A1, ..., An, V, D]`.

  Raises:
    TypeError: if the input types are invalid.
    ValueError: if the input dimensions are invalid.
  """
  with tf.compat.v1.name_scope(
      name, 'graph_convolution_edge_convolution_template',
      [data, neighbors, sizes, edge_function, edge_function_kwargs]):
    # Validate the input types and dimensions.
    data = tf.convert_to_tensor(value=data)
    neighbors = tf.compat.v1.convert_to_tensor_or_sparse_tensor(value=neighbors)
    if sizes is not None:
      sizes = tf.convert_to_tensor(value=sizes)
    utils.check_valid_graph_convolution_input(data, neighbors, sizes)

    # Flatten the batch dimensions and remove any vertex padding.
    data_ndims = data.shape.ndims
    if data_ndims > 2:
      if sizes is not None:
        sizes_square = tf.stack((sizes, sizes), axis=-1)
      else:
        sizes_square = None
      x_flat, unflatten = utils.flatten_batch_to_2d(data, sizes)
      adjacency = utils.convert_to_block_diag_2d(neighbors, sizes_square)
    else:
      x_flat = data
      adjacency = neighbors

    vertex_features = tf.gather(x_flat, adjacency.indices[:, 0])
    neighbor_features = tf.gather(x_flat, adjacency.indices[:, 1])

    edge_features = edge_function(
        vertex_features, neighbor_features, **edge_function_kwargs)

    features = utils.partition_sums_2d(
        edge_features, adjacency.indices[:, 0], adjacency.values)
    if data_ndims > 2:
      features = unflatten(features)

    return features

# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
