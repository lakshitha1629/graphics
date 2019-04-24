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
"""Tests for graph convolution ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

import tensorflow_graphics.geometry.representation.mesh.convolution.graph_convolution as gc
from tensorflow_graphics.util import test_case


def _dense_to_sparse(data):
  """Convert a numpy array to a tf.SparseTensor."""
  indices = np.where(data)
  return tf.SparseTensor(
      np.stack(indices, axis=-1), data[indices], dense_shape=data.shape)


def _dummy_data(batch_size, num_vertices, num_channels):
  """Create inputs for feature_steered_convolution."""
  if batch_size > 0:
    data = np.zeros(shape=(batch_size, num_vertices, num_channels),
                    dtype=np.float32)
    neighbors = _dense_to_sparse(np.tile(
        np.eye(num_vertices, dtype=np.float32), (batch_size, 1, 1)))
  else:
    data = np.zeros(shape=(num_vertices, num_channels), dtype=np.float32)
    neighbors = _dense_to_sparse(np.eye(num_vertices, dtype=np.float32))
  return data, neighbors


def _dummy_variables(in_channels, out_channels, num_weight_matrices):
  """Create variable substitutes for feature_steered_convolution."""
  var_u = tf.zeros(shape=(in_channels, num_weight_matrices))
  var_v = tf.zeros(shape=(in_channels, num_weight_matrices))
  var_c = tf.zeros(shape=(num_weight_matrices))
  var_w = tf.zeros(shape=(num_weight_matrices, in_channels, out_channels))
  var_b = tf.zeros(shape=(out_channels))
  return var_u, var_v, var_c, var_w, var_b


def _random_data(
    batch_size, num_vertices, num_channels, padding, only_self_edges,
    data_type=np.float32, neighbors_type=np.float32, sizes_type=np.int32):
  """Create random inputs for feature_steered_convolution."""
  def  _random_data_2d(num_vertices, num_channels, padding, only_self_edges):
    size = num_vertices if not padding else np.random.randint(
        low=1, high=num_vertices + 1)
    data = np.random.uniform(size=(size, num_channels)).astype(np.float32)
    if only_self_edges:
      neighbors = np.eye(size, dtype=np.float32)
    else:
      random = np.random.uniform(size=(size, size)).astype(np.float32)
      neighbors = np.maximum(
          np.where(random > 0.75, np.ones_like(random), np.zeros_like(random)),
          np.eye(size, dtype=np.float32))
      neighbors = neighbors / np.sum(neighbors, axis=1, keepdims=True)
    if padding:
      data = np.pad(data, ((0, num_vertices - size), (0, 0)), "constant")
      neighbors = np.pad(
          neighbors,
          ((0, num_vertices - size), (0, num_vertices - size)), "constant")
      return data, neighbors, size
    else:
      return data, neighbors

  if batch_size > 0:
    list_2d = [_random_data_2d(
        num_vertices, num_channels,
        padding, only_self_edges) for _ in range(batch_size)]
    data = np.stack([i[0] for i in list_2d], 0).astype(data_type)
    neighbors = np.stack([i[1] for i in list_2d], 0).astype(neighbors_type)
    if padding:
      sizes = np.stack([i[2] for i in list_2d], 0).astype(sizes_type)
      return data, _dense_to_sparse(neighbors), sizes
    else:
      return data, _dense_to_sparse(neighbors)
  else:
    if padding:
      raise ValueError("Padding only allowed with batched data.")
    data, neighbors = _random_data_2d(
        num_vertices, num_channels, False, only_self_edges)
    return data.astype(data_type), _dense_to_sparse(
        neighbors.astype(neighbors_type))


def _random_variables(
    in_channels, out_channels, num_weight_matrices, dtype=np.float32):
  """Create random variables for feature_steered_convolution."""
  def _random_constant(shape, dtype):
    return tf.constant(np.random.uniform(size=shape).astype(dtype))

  var_u = _random_constant([in_channels, num_weight_matrices], dtype)
  var_v = _random_constant([in_channels, num_weight_matrices], dtype)
  var_c = _random_constant([num_weight_matrices], dtype)
  var_w = _random_constant(
      [num_weight_matrices, in_channels, out_channels], dtype)
  var_b = _random_constant([out_channels], dtype)
  return var_u, var_v, var_c, var_w, var_b


class GraphConvolutionTestFeatureSteeredConvolutionLayerTests(
    test_case.TestCase):

  @parameterized.parameters(
      (1, 1, 1, 1, 1, False),
      (4, 2, 3, None, 5, False),
      (1, 2, 3, 4, 5, True),
  )
  def test_feature_steered_convolution_layer_exception_not_raised_shapes(
      self, batch_size, num_vertices, in_channels, out_channels,
      num_weight_matrices, translation_invariant):
    """Check if the convolution parameters and output have correct shapes."""
    data, neighbors = _dummy_data(batch_size, num_vertices, in_channels)
    name_scope = "test"
    if tf.executing_eagerly():
      layer = gc.FeatureSteeredConvolutionKerasLayer(
          translation_invariant=translation_invariant,
          num_weight_matrices=num_weight_matrices,
          num_output_channels=out_channels,
          name=name_scope)

    def _run_convolution():
      """Run the appropriate feature steered convolution layer."""
      if tf.executing_eagerly():
        try:
          output = layer(inputs=[data, neighbors], sizes=None)
        except Exception as e:  # pylint: disable=broad-except
          self.fail("Exception raised: %s" % str(e))
      else:
        try:
          output = gc.feature_steered_convolution_layer(
              data=data, neighbors=neighbors, sizes=None,
              translation_invariant=translation_invariant,
              num_weight_matrices=num_weight_matrices,
              num_output_channels=out_channels, name=None, var_name=name_scope)
        except Exception as e:  # pylint: disable=broad-except
          self.fail("Exception raised: %s" % str(e))
      return output

    output = _run_convolution()
    output_shape = output.shape.as_list()
    out_channels = in_channels if out_channels is None else out_channels
    self.assertEqual(output_shape[-1], out_channels)
    self.assertAllEqual(output_shape[:-1], data.shape[:-1])

    def _get_var_shape(var_name):
      """Get the shape of a variable by name."""
      if tf.executing_eagerly():
        trainable_variables = layer.trainable_variables
        for tv in trainable_variables:
          if tv.name == name_scope + "/" + var_name + ":0":
            return tv.shape.as_list()
        raise ValueError("Variable not found.")
      else:
        with tf.compat.v1.variable_scope(name_scope, reuse=True):
          variable = tf.compat.v1.get_variable(
              var_name, initializer=tf.constant(0))
          return variable.shape.as_list()

    self.assertAllEqual(
        _get_var_shape("u"), [in_channels, num_weight_matrices])
    self.assertAllEqual(
        _get_var_shape("c"), [num_weight_matrices])
    self.assertAllEqual(
        _get_var_shape("b"), [out_channels])
    self.assertAllEqual(
        _get_var_shape("w"),
        [num_weight_matrices, in_channels, out_channels])
    if not translation_invariant:
      self.assertAllEqual(
          _get_var_shape("v"), [in_channels, num_weight_matrices])

  def test_feature_steered_convolution_layer_training(self):
    """Test a simple training loop."""
    # Generate a small valid input for a simple training task.
    # Four corners of a square.
    data = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    neighbors_indices = np.array([[0, 0], [0, 1], [0, 3],
                                  [1, 0], [1, 1], [1, 2],
                                  [2, 1], [2, 2], [2, 3],
                                  [3, 0], [3, 2], [3, 3]])
    neighbors = tf.SparseTensor(
        neighbors_indices, np.ones(shape=(12)) / 3.0, dense_shape=(4, 4))
    # Desired output is arbitrary.
    labels = np.reshape([-1.0, -0.5, 0.5, 1.0], (-1, 1))
    num_training_iterations = 5

    if tf.executing_eagerly():
      with tf.GradientTape(persistent=True) as tape:
        layer = gc.FeatureSteeredConvolutionKerasLayer(
            translation_invariant=False, num_weight_matrices=1,
            num_output_channels=1)
        output = layer(inputs=[data, neighbors], sizes=None)
        loss = tf.nn.l2_loss(output - labels)

      trainable_variables = layer.trainable_variables
      for _ in range(num_training_iterations):
        grads = tape.gradient(loss, trainable_variables)
        tf.compat.v1.train.GradientDescentOptimizer(1e-4).apply_gradients(
            zip(grads, trainable_variables))
    else:
      output = gc.feature_steered_convolution_layer(
          data=data, neighbors=neighbors, sizes=None,
          translation_invariant=False, num_weight_matrices=1,
          num_output_channels=1)
      train_op = tf.compat.v1.train.GradientDescentOptimizer(1e-4).minimize(
          tf.nn.l2_loss(output - labels))
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.initialize_all_variables())
        for _ in range(num_training_iterations):
          sess.run(train_op)


class GraphConvolutionTestFeatureSteeredConvolutionTests(test_case.TestCase):

  @parameterized.parameters(
      ("'sizes' must have an integer type.",
       np.float32, np.float32, np.float32, np.float32),
      ("'data' must have a float type.",
       np.int32, np.float32, np.int32, np.float32),
      ("'neighbors' and 'data' must have the same type.",
       np.float32, np.float64, np.int32, np.float32),
  )
  def test_feature_steered_convolution_exception_raised_types(
      self, err_msg, data_type, neighbors_type, sizes_type, var_type):
    """Check the type errors for invalid input types."""
    data, neighbors, sizes = _random_data(
        1, 5, 3, True, False, data_type, neighbors_type, sizes_type)
    u, v, c, w, b = _random_variables(3, 3, 1, var_type)
    with self.assertRaisesRegexp(TypeError, err_msg):
      _ = gc.feature_steered_convolution(
          data=data, neighbors=neighbors, sizes=sizes,
          var_u=u, var_v=v, var_c=c, var_w=w, var_b=b)

  @parameterized.parameters(
      (np.float32, np.float32, np.int32, np.float32),
      (np.float64, np.float64, np.int32, np.float64),
      (np.float32, np.float32, np.int64, np.float32),
      (np.float64, np.float64, np.int64, np.float64),
  )
  def test_feature_steered_convolution_exception_not_raised_types(
      self, data_type, neighbors_type, sizes_type, var_type):
    """Check there are no exceptions for valid input types."""
    data, neighbors, sizes = _random_data(
        1, 5, 3, True, False, data_type, neighbors_type, sizes_type)
    u, v, c, w, b = _random_variables(3, 3, 1, var_type)
    try:
      gc.feature_steered_convolution(
          data=data, neighbors=neighbors, sizes=sizes,
          var_u=u, var_v=v, var_c=c, var_w=w, var_b=b)
    except Exception as e:  # pylint: disable=broad-except
      self.fail("Exception raised: %s" % str(e))

  def test_feature_steered_convolution_exception_raised_shapes(self):
    """Check that invalid input shapes trigger the right exceptions."""
    with self.assertRaisesRegexp(
        ValueError, "'data' and 'neighbors' must have the same rank."):
      data, neighbors = _dummy_data(1, 5, 2)
      u, v, c, w, b = _dummy_variables(2, 2, 1)
      data = data[0, :]
      _ = gc.feature_steered_convolution(
          data=data, neighbors=neighbors, sizes=None,
          var_u=u, var_v=v, var_c=c, var_w=w, var_b=b)

    with self.assertRaisesRegexp(
        ValueError, "'data' and 'neighbors' must have rank >= 2."):
      u, v, c, w, b = _dummy_variables(2, 2, 1)
      data = np.ones(shape=(5), dtype=np.float32)
      neighbors = _dense_to_sparse(np.ones(shape=(5), dtype=np.float32))
      _ = gc.feature_steered_convolution(
          data=data, neighbors=neighbors, sizes=None,
          var_u=u, var_v=v, var_c=c, var_w=w, var_b=b)

    with self.assertRaisesRegexp(
        ValueError,
        "'sizes' shape and the batch shape of 'data' must be equal."):
      data, neighbors = _dummy_data(1, 5, 2)
      u, v, c, w, b = _dummy_variables(2, 2, 1)
      _ = gc.feature_steered_convolution(
          data=data, neighbors=neighbors, sizes=((1, 1), (1, 1)),
          var_u=u, var_v=v, var_c=c, var_w=w, var_b=b)

  @parameterized.parameters(
      (1, 1, 1, 1, 1),
      (4, 2, 3, 6, 5),
      (0, 1, 1, 1, 1),
      (0, 2, 3, 6, 5),
  )
  def test_feature_steered_convolution_output_shape(
      self, batch_size, num_vertices, in_channels, out_channels,
      num_weight_matrices):
    """Check that the output of convolution has the correct shape."""
    data, neighbors = _dummy_data(batch_size, num_vertices, in_channels)
    u, v, c, w, b = _dummy_variables(
        in_channels, out_channels, num_weight_matrices)
    y = gc.feature_steered_convolution(
        data=data, neighbors=neighbors, sizes=None,
        var_u=u, var_v=v, var_c=c, var_w=w, var_b=b)
    y_shape = y.shape.as_list()
    self.assertEqual(y_shape[-1], out_channels)
    self.assertAllEqual(y_shape[:-1], data.shape[:-1])

  @parameterized.parameters(
      (1, 1, 1, 1, 1),
      (4, 2, 3, 6, 5),
      (0, 1, 1, 1, 1),
      (0, 2, 3, 6, 5),
  )
  def test_feature_steered_convolution_only_self_edges(
      self, batch_size, num_vertices, in_channels, out_channels,
      num_weight_matrices):
    """Test convolution when the graph only has self edges."""
    data, neighbors = _random_data(batch_size, num_vertices, in_channels,
                                   padding=False, only_self_edges=True)
    u, v, c, w, b = _random_variables(
        in_channels, out_channels, num_weight_matrices)

    with self.subTest(name="w=0_expect_output=b"):
      # If w = 0, then y = b.
      y = gc.feature_steered_convolution(
          data=data, neighbors=neighbors, sizes=None,
          var_u=u, var_v=v, var_c=c, var_w=tf.zeros_like(w),
          var_b=b)
      y_expected = tf.broadcast_to(b, y.shape)
      self.assertAllEqual(y, y_expected)

    with self.subTest(name="translation_invariant_self_edges"):
      # u = -v, and graph only has self-edges.
      y = gc.feature_steered_convolution(
          data=data, neighbors=neighbors, sizes=None,
          var_u=u, var_v=-u, var_c=c, var_w=w, var_b=b)
      q = tf.reshape(tf.exp(c) / tf.reduce_sum(input_tensor=tf.exp(c)),
                     (num_weight_matrices, 1, 1))
      if batch_size > 0:
        q_times_w = tf.reduce_sum(input_tensor=q * w, axis=0, keepdims=True)
        q_times_w = tf.tile(q_times_w, (batch_size, 1, 1))
      else:
        q_times_w = tf.reduce_sum(input_tensor=q * w, axis=0)
      y_expected = tf.matmul(data, q_times_w) + tf.broadcast_to(b, y.shape)
      self.assertAllClose(y, y_expected)

    with self.subTest(name="constant_signal"):
      # Convolving a constant signal results in a constant signal.
      if batch_size > 0:
        constant_data = np.tile(np.random.uniform(
            size=(batch_size, 1, in_channels)).astype(
                np.float32), (1, num_vertices, 1))
      else:
        constant_data = np.tile(np.random.uniform(
            size=(1, in_channels)).astype(np.float32), (num_vertices, 1))
      y = gc.feature_steered_convolution(
          data=constant_data, neighbors=neighbors, sizes=None,
          var_u=u, var_v=v, var_c=c, var_w=w, var_b=b)
      if batch_size > 0:
        y_expected = tf.tile(y[:, :1, :], (1, num_vertices, 1))
      else:
        y_expected = tf.tile(y[:1, :], (num_vertices, 1))
      self.assertAllClose(y, y_expected)

  @parameterized.parameters(
      ([[1.0], [2.0], [3.0]], np.ones(shape=(3, 3)) / 3.0, [[0.5]],
       [[1.3]], [-0.7], [[[0.8]]], [3.0], [[4.6], [4.6], [4.6]]),
      ([[1.0], [2.0], [3.0]], np.ones(shape=(3, 3)) / 3.0, [[0.5, 0.2]],
       [[0.3, 0.4]], [-0.7, 0.15], [[[0.8]], [[1.1]]], [3.0],
       [[5.011706928844621], [4.971030281984818], [4.927388658982911]]),
  )
  def test_feature_steered_convolution_padding_preset(
      self, data, neighbors, u, v, c, w, b, expected):
    """Test expected result for preset data and filter values."""
    array = (np.array(i) for i in (data, neighbors, expected))
    data, neighbors, expected = array
    tensors = (tf.convert_to_tensor(value=np.array(i).astype(data.dtype)) \
               for i in (u, v, c, w, b))
    u, v, c, w, b = tensors
    y = gc.feature_steered_convolution(
        data=data, neighbors=_dense_to_sparse(neighbors), sizes=None,
        var_u=u, var_v=v, var_c=c, var_w=w, var_b=b)
    self.assertAllClose(y, expected)

  @parameterized.parameters(
      (1, 5, 1, 1, 1),
      (2, 6, 3, 6, 5),
      (5, 15, 6, 12, 8),
  )
  def test_feature_steered_convolution_padding_random(
      self, batch_size, num_vertices, in_channels, out_channels,
      num_weight_matrices):
    """Test mixed topology batches (random vertices and neighbors)."""
    data, neighbors, sizes = _random_data(batch_size, num_vertices, in_channels,
                                          padding=True, only_self_edges=False)
    u, v, c, w, b = _random_variables(
        in_channels, out_channels, num_weight_matrices)

    # If w = 0, then y = b.
    y = gc.feature_steered_convolution(
        data=data, neighbors=neighbors, sizes=sizes,
        var_u=u, var_v=v, var_c=c, var_w=tf.zeros_like(w),
        var_b=b)
    for k in range(batch_size):
      y_crop = y[k, :sizes[k], :]
      y_expected = tf.broadcast_to(b, y_crop.shape)
      self.assertAllEqual(y_crop, y_expected)
      # Check for zeros in the padded region.
      self.assertAllEqual(y[k, sizes[k]:, :],
                          tf.zeros((num_vertices - sizes[k], out_channels)))

    # Convolving a constant signal results in a constant signal.
    constant_data = data
    for k in range(batch_size):
      constant_data[k, :sizes[k], :] = np.tile(data[k, 0, :], (sizes[k], 1))
    y = gc.feature_steered_convolution(
        data=constant_data, neighbors=neighbors, sizes=sizes,
        var_u=u, var_v=v, var_c=c, var_w=w, var_b=b)
    for k in range(batch_size):
      y_crop = y[k, :sizes[k], :]
      y_const = tf.broadcast_to(y_crop[0, :], y_crop.shape)
      self.assertAllClose(y_crop, y_const)
      # Check for zeros in the padded region.
      self.assertAllEqual(y[k, sizes[k]:, :],
                          tf.zeros([num_vertices - sizes[k], out_channels]))

  @parameterized.parameters(
      (1, 10, 3, 1, True), (3, 6, 1, 4, True), (0, 10, 5, 2, False),
      (1, 10, 3, 1, False), (3, 6, 1, 4, False), (0, 10, 5, 2, False),
  )
  def test_feature_steered_convolution_jacobian_random(
      self, batch_size, num_vertices, in_channels, num_weight_matrices,
      padding):
    """Test the jacobian for random input data."""
    random_data = _random_data(
        batch_size, num_vertices, in_channels, padding,
        only_self_edges=False, data_type=np.float64, neighbors_type=np.float64)
    data_init = random_data[0]
    neighbors = random_data[1]
    sizes = None if not padding else random_data[2]
    u, v, c, w, b = _random_variables(
        in_channels, in_channels, num_weight_matrices, dtype=np.float64)
    data = tf.convert_to_tensor(value=data_init)
    y = gc.feature_steered_convolution(
        data=data, neighbors=neighbors, sizes=sizes,
        var_u=u, var_v=v, var_c=c, var_w=w, var_b=b)
    self.assert_jacobian_is_correct(data, data_init, y)

  @parameterized.parameters(
      (1, 1, 0.0), (5, 1, 0.0), (1, 3, 0.0), (5, 3, 0.0),
      (1, 1, 1.0), (5, 1, 1.0), (1, 3, 1.0), (5, 3, 1.0),
  )
  def test_feature_steered_convolution_jacobian_preset(
      self, num_vertices, num_channels, data_multiplier):
    """Test the jacobian is correct for preset inputs."""
    # Corner cases include one vertex, one channel, and all-zero features.
    data_init = data_multiplier * np.random.uniform(
        size=(num_vertices, num_channels)).astype(np.float64)
    neighbors = tf.sparse.eye(num_vertices, dtype=tf.float64)
    u, v, c, w, b = _random_variables(
        num_channels, num_channels, 1, dtype=np.float64)
    data = tf.convert_to_tensor(value=data_init)
    y = gc.feature_steered_convolution(
        data=data, neighbors=neighbors, sizes=None,
        var_u=u, var_v=v, var_c=c, var_w=w,
        var_b=b)
    self.assert_jacobian_is_correct(data, data_init, y)


class EdgeConvolutionTemplateTests(test_case.TestCase):

  def _zeros(self, vertex_features, neighbor_features, out_dimensions=None):
    """A callable for `edge_convolution_template`."""
    if out_dimensions is None:
      return tf.zeros_like(vertex_features)
    else:
      return tf.zeros(
          shape=(vertex_features.shape.as_list()[0], out_dimensions),
          dtype=vertex_features.dtype)

  def _pass_through(self, vertex_features, neighbor_features):
    """A callable for `edge_convolution_template`."""
    return neighbor_features

  def _circular_2d_data(self, num_vertices, include_normals=False):
    """Create data for a circle graph."""
    # Vertices are points distributed uniformly on a circle, with each point
    # connected to its closest neighbor on either side.
    theta = np.linspace(0.0, np.pi * 2.0, num=num_vertices, endpoint=False)
    data = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    if include_normals:
      data = np.concatenate((data, data), axis=-1)
    eye = np.eye(num_vertices)
    neighbors = np.maximum(np.roll(eye, 1, axis=1),
                           np.roll(eye, -1, axis=1)) * 0.5
    return data, _dense_to_sparse(neighbors)

  def _edge_curvature_2d(self, vertex_features, neighbor_features):
    """A callable for `edge_convolution_template` that computes curvature."""
    x_position, x_normal = tf.split(
        value=vertex_features, num_or_size_splits=2, axis=-1)
    y_position, y_normal = tf.split(
        value=neighbor_features, num_or_size_splits=2, axis=-1)
    yx_diff = x_position - y_position
    curvature_unscaled = tf.abs(
        tf.reduce_sum(input_tensor=(y_normal - x_normal) * yx_diff,
                      axis=-1, keepdims=True))
    edge_length_squared = tf.reduce_sum(
        input_tensor=yx_diff * yx_diff, axis=-1, keepdims=True)
    return tf.where(tf.less(edge_length_squared, 1e-7),
                    tf.zeros_like(edge_length_squared),
                    curvature_unscaled / edge_length_squared)

  @parameterized.parameters(
      ("'sizes' must have an integer type.",
       np.float32, np.float32, np.float32),
      ("'data' must have a float type.",
       np.int32, np.float32, np.int32),
      ("'neighbors' and 'data' must have the same type.",
       np.float32, np.float64, np.int32),
  )
  def test_edge_convolution_template_exception_raised_types(
      self, err_msg, data_type, neighbors_type, sizes_type):
    """Check the type errors for invalid input types."""
    data, neighbors, sizes = _random_data(
        1, 5, 3, True, False, data_type, neighbors_type, sizes_type)
    with self.assertRaisesRegexp(TypeError, err_msg):
      gc.edge_convolution_template(
          data=data, neighbors=neighbors, sizes=sizes,
          edge_function=self._zeros, edge_function_kwargs=dict())

  @parameterized.parameters(
      (np.float32, np.float32, np.int32),
      (np.float64, np.float64, np.int32),
      (np.float32, np.float32, np.int64),
      (np.float64, np.float64, np.int64),
      (np.float64, np.float64, np.int8),
      (np.float64, np.float64, np.uint8),
      (np.float64, np.float64, np.int16),
      (np.float64, np.float64, np.uint16),
  )
  def test_edge_convolution_template_exception_not_raised_types(
      self, data_type, neighbors_type, sizes_type):
    """Check there are no exceptions for valid input types."""
    data, neighbors, sizes = _random_data(
        1, 5, 3, True, False, data_type, neighbors_type, sizes_type)
    try:
      gc.edge_convolution_template(
          data=data, neighbors=neighbors, sizes=sizes,
          edge_function=self._zeros, edge_function_kwargs=dict())
    except Exception as e:  # pylint: disable=broad-except
      self.fail("Exception raised: %s" % str(e))

  def test_edge_convolution_template_exception_raised_shapes(self):
    """Check that invalid input shapes trigger the right exceptions."""
    with self.assertRaisesRegexp(
        ValueError, "'data' and 'neighbors' must have the same rank."):
      data, neighbors = _dummy_data(1, 5, 2)
      data = data[0, :]
      _ = gc.edge_convolution_template(
          data=data, neighbors=neighbors, sizes=None,
          edge_function=self._zeros, edge_function_kwargs=dict())

    with self.assertRaisesRegexp(
        ValueError, "'data' and 'neighbors' must have rank >= 2."):
      data = np.ones(shape=(5), dtype=np.float32)
      neighbors = _dense_to_sparse(np.ones(shape=(5), dtype=np.float32))
      _ = gc.edge_convolution_template(
          data=data, neighbors=neighbors, sizes=None,
          edge_function=self._zeros, edge_function_kwargs=dict())

    with self.assertRaisesRegexp(
        ValueError,
        "'sizes' shape and the batch shape of 'data' must be equal."):
      data, neighbors = _dummy_data(1, 5, 2)
      _ = gc.edge_convolution_template(
          data=data, neighbors=neighbors, sizes=((1, 1), (1, 1)),
          edge_function=self._zeros, edge_function_kwargs=dict())

  @parameterized.parameters(
      (1, 1, 1, 1),
      (4, 2, 3, 6),
      (0, 1, 1, 1),
      (0, 2, 3, 6),
  )
  def test_edge_convolution_template_output_shape(
      self, batch_size, num_vertices, in_channels, out_channels):
    """Check that the output of convolution has the correct shape."""
    data, neighbors = _dummy_data(batch_size, num_vertices, in_channels)
    y = gc.edge_convolution_template(
        data, neighbors, None, self._zeros,
        edge_function_kwargs={"out_dimensions": out_channels})
    y_shape = y.shape.as_list()
    with self.subTest(name="out_channels"):
      self.assertEqual(y_shape[-1], out_channels)
    with self.subTest(name="shape"):
      self.assertAllEqual(y_shape[:-1], data.shape[:-1])

  @parameterized.parameters(
      (1, 10, 3, True), (3, 6, 1, True), (0, 10, 5, False),
      (1, 10, 3, False), (3, 6, 1, False), (0, 10, 5, False),
  )
  def test_edge_convolution_template_jacobian_random(
      self, batch_size, num_vertices, in_channels, padding):
    """Test the jacobian for random input data."""
    random_data = _random_data(
        batch_size, num_vertices, in_channels, padding,
        only_self_edges=False, data_type=np.float64, neighbors_type=np.float64)
    data_init = random_data[0]
    neighbors = random_data[1]
    sizes = None if not padding else random_data[2]
    data = tf.convert_to_tensor(value=data_init)
    y = gc.edge_convolution_template(
        data=data, neighbors=neighbors, sizes=sizes,
        edge_function=self._pass_through, edge_function_kwargs=dict())
    self.assert_jacobian_is_correct(data, data_init, y)

  @parameterized.parameters(
      (1, 1, 0.0), (5, 1, 0.0), (1, 3, 0.0), (5, 3, 0.0),
      (1, 1, 1.0), (5, 1, 1.0), (1, 3, 1.0), (5, 3, 1.0),
  )
  def test_edge_convolution_template_jacobian_preset(
      self, num_vertices, num_channels, data_multiplier):
    """Test the jacobian is correct for preset inputs."""
    # Corner cases include one vertex, one channel, and all-zero features.
    data_init = data_multiplier * np.random.uniform(
        size=(num_vertices, num_channels)).astype(np.float64)
    neighbors = tf.sparse.eye(num_vertices, dtype=tf.float64)
    data = tf.convert_to_tensor(value=data_init)
    y = gc.edge_convolution_template(
        data=data, neighbors=neighbors, sizes=None,
        edge_function=self._pass_through, edge_function_kwargs=dict())
    self.assert_jacobian_is_correct(data, data_init, y)

  def test_edge_convolution_template_laplacian_smoothing(self):
    r"""Test the expected result with laplacian smoothing.

      Laplacian smoothing for meshes is defined as
      $$y_i = \frac{1}{|\mathcal{N(i)}|} \sum_{j \in \mathcal{N(i)}} x_j$$

      This can be computed using `edge_convolution_template` with `f(x, y)->y`.
    """

    # We can reuse `self._pass_through(x, y)->y` as the smoothing functional.
    with self.subTest(name="only_self_edges_random"):
      num_vertices = 500
      data = np.random.uniform(size=(num_vertices, 5))
      neighbors = tf.sparse.eye(num_vertices, dtype=tf.as_dtype(data.dtype))
      data_smoothed = gc.edge_convolution_template(
          data=data, neighbors=neighbors, sizes=None,
          edge_function=self._pass_through, edge_function_kwargs=dict())
      self.assertAllEqual(data, data_smoothed)

    with self.subTest(name="circular_2d"):
      num_vertices = 500
      data, neighbors = self._circular_2d_data(num_vertices)
      data_smoothed = gc.edge_convolution_template(
          data=data, neighbors=neighbors, sizes=None,
          edge_function=self._pass_through, edge_function_kwargs=dict())
      # The smoothed points should have the same direction as the originals.
      data_smoothed_normalized = tf.nn.l2_normalize(data_smoothed, axis=-1)
      self.assertAllClose(data, data_smoothed_normalized)

  def test_edge_convolution_template_curvature(self):
    r"""Test the expected result with curvature.

      (Approximate) curvature for meshes is defined as
      $$\kappa_{v_i} = \frac{1}{|\mathcal{N}(v_i)|}
        \sum_{v_j \in \mathcal{N}(v_i)}
        \frac{(\vec{v_i} - \vec{v_j})^T (\vec{n_{v_i}} -
        \vec{n_{v_j}})} {\left|\vec{v_i}-\vec{v_j}\right|^2}
      $$

      This can be computed using `edge_convolution_template` with
        $$f(x, y) = (n_x - n_y)^T (x - y) / ||x - y||^2.$$
      where $$n_x$$ and $$n_y$$ are the normals at points $$x$$ and $$y$$
      respectively.
    """
    # We can reuse `self._edge_curvature_2d` as the curvature functional.
    num_vertices = 500
    data, neighbors = self._circular_2d_data(num_vertices,
                                             include_normals=True)
    data_curvature = gc.edge_convolution_template(
        data=data, neighbors=neighbors, sizes=None,
        edge_function=self._edge_curvature_2d, edge_function_kwargs=dict())
    # The curvature at each point on a circle of radius 1 should be 1.
    self.assertAllClose(data_curvature, np.ones(shape=(num_vertices, 1)))


if __name__ == "__main__":
  test_case.main()
