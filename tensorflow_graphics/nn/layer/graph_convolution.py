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
"""This module implements graph convolutions layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow_graphics.geometry.convolution.graph_convolution as gc
from tensorflow_graphics.util import export_api


def feature_steered_convolution_layer(
    data,
    neighbors,
    sizes,
    translation_invariant=True,
    num_weight_matrices=8,
    num_output_channels=None,
    initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1),
    name=None,
    var_name=None):
  """Wraps the function `feature_steered_convolution` as a TensorFlow layer.

  The shorthands used below are
    `V`: The number of vertices.
    `C`: The number of channels in the input data.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    data: A `float` tensor with shape `[A1, ..., An, V, C]`.
    neighbors: A SparseTensor with the same type as `data` and with shape `[A1,
      ..., An, V, V]` representing vertex neighborhoods. The neighborhood of a
      vertex defines the support region for convolution. For a mesh, a common
      choice for the neighborhood of vertex i would be the vertices in the
      K-ring of i (including i itself). Each vertex must have at least one
      neighbor. For a faithful implementation of the FeaStNet paper, neighbors
      should be a row-normalized weight matrix
      corresponding to the graph adjacency matrix with self-edges:
        `neighbors[A1, ..., An, i, j] > 0` if vertex j is a neighbor of vertex
        i, and `neighbors[A1, ..., An, i, i] > 0` for all i, and `sum(neighbors,
        axis=-1)[A1, ..., An, i] == 1.0 for all i`. These requirements are
        relaxed in this implementation.
    sizes: An `int` tensor of shape `[A1, ..., An]` indicating the true input
      sizes in case of padding (`sizes=None` indicates no padding). `sizes[A1,
      ..., An] <= V`. If `data` and `neighbors` are 2-D, `sizes` will
      be ignored. An example usage of `sizes`: consider an input consisting of
        three graphs G0, G1, and G2 with V0, V1, and V2 vertices respectively.
      The padded input would have the following shapes: `data.shape = [3, V,
        C]`, and `neighbors.shape = [3, V, V]`, where `V=max([V0, V1, V2])`. The
        true sizes of each graph will be specified by
      `sizes=[V0, V1, V2]`, and `data[i, :Vi, :]` and `neighbors[i, :Vi, :Vi]`
        will be the vertex and neighborhood data of graph Gi. The SparseTensor
        `neighbors` should have no nonzero entries in the padded regions.
    translation_invariant: A `bool`. If `True` the assignment of features to
      weight matrices will be invariant to translation.
    num_weight_matrices: An `int` specifying the number of weight matrices used
      in the convolution.
    num_output_channels: An optional `int` specifying the number of channels in
      the output. If `None` then `num_output_channels = C`.
    initializer: An initializer for the trainable variables.
    name: A (name_scope) name for this op. Passed through to
      feature_steered_convolution().
    var_name: A (var_scope) name for the variables. Defaults to
      `graph_convolution_feature_steered_convolution_weights`.

  Returns:
    Tensor with shape `[A1, ..., An, V, num_output_channels]`.
  """
  with tf.compat.v1.variable_scope(
      var_name,
      default_name='graph_convolution_feature_steered_convolution_weights'):
    # Skips shape validation to avoid redundancy with
    # feature_steered_convolution().
    data = tf.convert_to_tensor(value=data)
    in_channels = data.get_shape().as_list()[-1]
    if num_output_channels is None:
      out_channels = in_channels
    else:
      out_channels = num_output_channels

    var_u = tf.compat.v1.get_variable(
        shape=(in_channels, num_weight_matrices),
        dtype=data.dtype,
        initializer=initializer,
        name='u')
    if translation_invariant:
      var_v = -var_u
    else:
      var_v = tf.compat.v1.get_variable(
          shape=(in_channels, num_weight_matrices),
          dtype=data.dtype,
          initializer=initializer,
          name='v')
    var_c = tf.compat.v1.get_variable(
        shape=(num_weight_matrices),
        dtype=data.dtype,
        initializer=initializer,
        name='c')
    var_w = tf.compat.v1.get_variable(
        shape=(num_weight_matrices, in_channels, out_channels),
        dtype=data.dtype,
        initializer=initializer,
        name='w')
    var_b = tf.compat.v1.get_variable(
        shape=(out_channels),
        dtype=data.dtype,
        initializer=initializer,
        name='b')

    return gc.feature_steered_convolution(
        data=data,
        neighbors=neighbors,
        sizes=sizes,
        var_u=var_u,
        var_v=var_v,
        var_c=var_c,
        var_w=var_w,
        var_b=var_b,
        name=name)


class FeatureSteeredConvolutionKerasLayer(tf.keras.layers.Layer):
  """Wraps the function `feature_steered_convolution` as a Keras layer."""

  def __init__(self,
               translation_invariant=True,
               num_weight_matrices=8,
               num_output_channels=None,
               initializer=None,
               name=None,
               **kwargs):
    """Initializes FeatureSteeredConvolutionKerasLayer.

    Args:
      translation_invariant: A `bool`. If `True` the assignment of features to
        weight matrices will be invariant to translation.
      num_weight_matrices: An `int` specifying the number of weight matrices
        used in the convolution.
      num_output_channels: An optional `int` specifying the number of channels
        in the output. If `None` then `num_output_channels` will be the same as
        the input dimensionality.
      initializer: An initializer for the trainable variables. If `None`,
        defaults to tf.compat.v1.truncated_normal_initializer(stddev=0.1).
      name: A name for this layer.
      **kwargs: Additional keyword arguments passed to the base layer.
    """
    super(FeatureSteeredConvolutionKerasLayer, self).__init__(
        name=name, **kwargs)
    self._num_weight_matrices = num_weight_matrices
    self._num_output_channels = num_output_channels
    self._translation_invariant = translation_invariant
    if initializer is None:
      self._initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.1)
    else:
      self._initializer = initializer

  def build(self, input_shape):
    """Initializes the trainable weights."""
    in_channels = tf.TensorShape(input_shape[0]).as_list()[-1]
    if self._num_output_channels is None:
      out_channels = in_channels
    else:
      out_channels = self._num_output_channels
    dtype = self.dtype
    num_weight_matrices = self._num_weight_matrices
    initializer = self._initializer

    self.var_u = self.add_weight(
        shape=(in_channels, num_weight_matrices),
        dtype=dtype,
        initializer=initializer,
        name='u')
    if self._translation_invariant:
      self.var_v = -self.var_u
    else:
      self.var_v = self.add_weight(
          shape=(in_channels, num_weight_matrices),
          dtype=dtype,
          initializer=initializer,
          name='v')
    self.var_c = self.add_weight(
        shape=(num_weight_matrices,),
        dtype=dtype,
        initializer=initializer,
        name='c')
    self.var_w = self.add_weight(
        shape=(num_weight_matrices, in_channels, out_channels),
        dtype=dtype,
        initializer=initializer,
        name='w',
        trainable=True)
    self.var_b = self.add_weight(
        shape=(out_channels,),
        dtype=dtype,
        initializer=initializer,
        name='b',
        trainable=True)

  def call(self, inputs, sizes=None):
    """Executes the convolution.

    The shorthands used below are
      `V`: The number of vertices.
      `C`: The number of channels in the input data.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      inputs: A list of two tensors `[data, neighbors]`. `data` is a `float`
        tensor with shape `[A1, ..., An, V, C]`. `neighbors` is a SparseTensor
        with the same type as `data` and with shape `[A1, ..., An, V, V]`
        representing vertex neighborhoods. The neighborhood of a vertex defines
        the support region for convolution. For a mesh, a common choice for the
        neighborhood of vertex i would be the vertices in the K-ring of i
        (including i itself). Each vertex must have at least one neighbor. For a
        faithful implementation of the FeaStNet paper, neighbors should be a
        row-normalized weight matrix corresponding to the graph adjacency
        matrix with self-edges: `neighbors[A1, ..., An, i, j] > 0` if vertex `j`
          is a neighbor of vertex `i`, and `neighbors[A1, ..., An, i, i] > 0`
          for all `i`, and `sum(neighbors, axis=-1)[A1, ..., An, i] == 1.0 for
          all i`. These requirements are relaxed in this implementation.
      sizes: An `int` tensor of shape `[A1, ..., An]` indicating the true input
        sizes in case of padding (`sizes=None` indicates no padding). `sizes[A1,
        ..., An] <= V`. If `data` and `neighbors` are 2-D, `sizes`
        will be ignored. An example usage of `sizes`: consider an input
          consisting of three graphs G0, G1, and G2 with V0, V1, and V2 vertices
        respectively. The padded input would have the following shapes:
          `data.shape = [3, V, C]`, and `neighbors.shape = [3, V, V]`, where
          `V=max([V0, V1, V2])`. The true sizes of each graph will be specified
          by
        `sizes=[V0, V1, V2]`, and `data[i, :Vi, :]` and `neighbors[i, :Vi, :Vi]`
          will be the vertex and neighborhood data of graph Gi. The SparseTensor
          `neighbors` should have no nonzero entries in the padded regions.

    Returns:
      Tensor with shape `[A1, ..., An, V, num_output_channels]`.
    """
    return gc.feature_steered_convolution(
        data=inputs[0],
        neighbors=inputs[1],
        sizes=sizes,
        var_u=self.var_u,
        var_v=self.var_v,
        var_c=self.var_c,
        var_w=self.var_w,
        var_b=self.var_b)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
