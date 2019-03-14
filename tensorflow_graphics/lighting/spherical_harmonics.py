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
"""This module implements routines required for spherical harmonics lighting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import math


def integration_product(f, g, keepdims=True, name=None):
  """Computes the integral of f.g over the sphere.

  Args:
    f: N-D tensor of shape `[?, ..., ?]`. Stores spherical harmonics
      coefficients in the last dimension.
    g: N-D tensor of shape `[?, ..., ?]`. Stores spherical harmonics
      coefficients in the last dimension.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for this op. Defaults to "spherical_harmonics_convolution".

  Returns:
    (N-1)-D tensor of shape `[?, ..., ?]`.

  Raises:
    ValueError: if the last dimension of `f` is different from the last
    dimension of `g`.
  """
  with tf.compat.v1.name_scope(name, "spherical_harmonics_convolution", [f, g]):
    f = tf.convert_to_tensor(value=f)
    g = tf.convert_to_tensor(value=g)
    if f.shape.as_list()[-1] != g.shape.as_list()[-1]:
      raise ValueError("'f' and 'g' differ in their last dimension.")
    return vector.dot(f, g, keepdims=keepdims)


def generate_l_m_permutations(max_band, name=None):
  """Generate l and m coefficients for spherical harmonics.

  Args:
    max_band: An integer scalar storing the highest band.
    name: A name for this op. Defaults to
      'spherical_harmonics_generate_l_m_permutations'.

  Returns:
    Two tensors of shape `[max_band*max_band]`.
  """
  with tf.compat.v1.name_scope(
      name, "spherical_harmonics_generate_l_m_permutations", [max_band]):
    l = []
    m = []
    for l_ in range(0, max_band + 1):
      for m_ in range(-l_, l_ + 1):
        l.append(l_)
        m.append(m_)
    return tf.convert_to_tensor(value=l), tf.convert_to_tensor(value=m)


def generate_l_m_zonal(max_band, name=None):
  """Generate l and m coefficients for zonal harmonics.

  Args:
    max_band: An integer scalar storing the highest band.
    name: A name for this op. Defaults to
      'spherical_harmonics_generate_l_m_zonal'.

  Returns:
    Two tensors of shape `[max_band+1]`.
  """
  with tf.compat.v1.name_scope(name, "generate_l_m_zonal", [max_band]):
    l = np.linspace(0, max_band, num=max_band + 1)
    m = np.zeros(max_band + 1)
    return tf.convert_to_tensor(value=l), tf.convert_to_tensor(value=m)


def _evaluate_legendre_polynomial_pmm_eval(m, x):
  pmm = tf.pow(1.0 - tf.pow(x, 2.0), tf.cast(m, dtype=x.dtype) / 2.0)
  ones = tf.ones_like(m)
  pmm *= tf.cast(
      tf.pow(-ones, m) * math.double_factorial(2 * m - 1), dtype=pmm.dtype)
  return pmm


# pylint: disable=unused-argument
def _evaluate_legendre_polynomial_loop_cond(x, n, l, m, pmm, pmm1):
  return tf.cast(tf.math.count_nonzero(n <= l), tf.bool)


def _evaluate_legendre_polynomial_loop_body(x, n, l, m, pmm, pmm1):
  n_float = tf.cast(n, dtype=x.dtype)
  m_float = tf.cast(m, dtype=x.dtype)
  pmn = (x * (2.0 * n_float - 1.0) * pmm1 - (n_float + m_float - 1) * pmm) / (
      n_float - m_float)
  pmm = tf.where(tf.less_equal(n, l), pmm1, pmm)
  pmm1 = tf.where(tf.less_equal(n, l), pmn, pmm1)
  n += 1
  return x, n, l, m, pmm, pmm1


def _evaluate_legendre_polynomial_loop(x, m, l, pmm, pmm1):
  n = m + 2
  x, n, l, m, pmm, pmm1 = tf.while_loop(
      cond=_evaluate_legendre_polynomial_loop_cond,
      body=_evaluate_legendre_polynomial_loop_body,
      loop_vars=[x, n, l, m, pmm, pmm1])
  return pmm1


def _evaluate_legendre_polynomial_branch(l, m, x, pmm):
  pmm1 = x * (2.0 * tf.cast(m, dtype=x.dtype) + 1.0) * pmm
  # if, l == m + 1 return pmm1, otherwise lift to the next band.
  res = tf.where(
      tf.equal(l, m + 1), pmm1,
      _evaluate_legendre_polynomial_loop(x, m, l, pmm, pmm1))
  return res


def evaluate_legendre_polynomial(l, m, x):
  """Evaluates the Legendre polynomial of degree l and order m at x.

  Note:
    Followed the implementation p. 10 of `Spherical Harmonic Lighting: The
    Gritty Details`.

  Args:
    l: N-D tensor of shape `[?, ..., ?]` with l >= 0.
    m: N-D tensor of shape `[?, ..., ?]` with 0 <= m <= l.
    x: N-D tensor of shape `[?, ..., ?]` with values in [-1,1].

  Returns:
    N-D tensor of shape `[?, ..., ?]`.
  """
  # Conversion to tensors.
  l = tf.convert_to_tensor(value=l)
  m = tf.convert_to_tensor(value=m)
  x = tf.convert_to_tensor(value=x)
  # Checks that the input is in the appropriate range.
  l = asserts.assert_all_above(l, 0)
  m = asserts.assert_all_in_range(m, 0, l)
  x = asserts.assert_all_in_range(x, -1.0, 1.0)
  pmm = _evaluate_legendre_polynomial_pmm_eval(m, x)
  # if l == m return pmm.
  res = tf.where(
      tf.equal(l, m), pmm, _evaluate_legendre_polynomial_branch(l, m, x, pmm))
  return res


def _spherical_harmonics_normalization(l, m, var_type=tf.float64):
  l = tf.convert_to_tensor(value=l)
  m = tf.convert_to_tensor(value=m)
  l = tf.cast(l, dtype=var_type)
  m = tf.cast(m, dtype=var_type)
  numerator = (2.0 * l + 1.0) * math.factorial(l - tf.abs(m))
  denominator = 4.0 * np.pi * math.factorial(l + tf.abs(m))
  return tf.sqrt(numerator / denominator)


# pylint: disable=missing-docstring
def _evaluate_spherical_harmonics_branch(degree,
                                         order,
                                         theta,
                                         phi,
                                         sign_order,
                                         var_type=tf.float64):
  sqrt_2 = tf.constant(1.41421356237, dtype=var_type)
  order_float = tf.cast(order, dtype=var_type)
  tmp = sqrt_2 * _spherical_harmonics_normalization(
      degree, order, var_type) * evaluate_legendre_polynomial(
          degree, order, tf.cos(theta))
  positive = tmp * tf.cos(order_float * phi)
  negative = tmp * tf.sin(order_float * phi)
  res = tf.where(tf.greater(sign_order, 0), positive, negative)
  return res


def evaluate_spherical_harmonics(l, m, theta, phi, name=None):
  """Evaluates a point sample of a Spherical Harmonic basis function.

  Args:
    l: N-D tensor of shape `[?, ..., ?]`. This variable must contain
      non-negative integer values.
    m: N-D tensor of shape `[?, ..., ?]`. This varibles must contain integer
      values in [-l, l].
    theta: N-D tensor of shape `[?, ..., ?, 1]`. This variable stores
      the polar angle. Values of theta must be in [0, pi].
    phi: N-D tensor of shape `[?, ..., ?, 1]`. This variable stores the
      azimuthal angle. Values of phi must be in [0, 2pi].
    name: A name for this op. Defaults to
    'spherical_harmonics_evaluate_spherical_harmonics'.

  Note: Followed the implementation and variable names used p. 12 of 'Spherical
    Harmonic Lighting: The Gritty Details'.

  Returns:
    N-D tensor of shape `[?, ..., ?]`.

  Raises:
    ValueError: if the shape of `theta` or `phi` is not supported.
    InvalidArgumentError: if at least an element of `l`, `m`, `theta` or `phi`
    is outside the expected range.
  """
  with tf.compat.v1.name_scope(
      name, "spherical_harmonics_evaluate_spherical_harmonics",
      [l, m, theta, phi]):
    # Conversion to tensors.
    l = tf.convert_to_tensor(value=l)
    m = tf.convert_to_tensor(value=m)
    theta = tf.convert_to_tensor(value=theta)
    phi = tf.convert_to_tensor(value=phi)
    # Checks that tensors have the appropriate shape.
    l_shape = l.shape.as_list()
    theta_shape = theta.shape.as_list()
    if l_shape != m.shape.as_list():
      raise ValueError("'l' and 'm' have different shapes.")
    if theta_shape != phi.shape.as_list():
      raise ValueError("'theta' and 'phi' have different shapes.")
    if l_shape[:-1] != theta_shape[:-1]:
      raise ValueError(
          "'l', 'm', 'theta', and 'phi' must have the same shape in all but the last dimension."
      )
    # Checks that tensors contain appropriate data.
    l = asserts.assert_all_above(l, 0)
    m = asserts.assert_all_in_range(m, -l, l)
    theta = asserts.assert_all_in_range(theta, 0.0, np.pi)
    phi = asserts.assert_all_in_range(phi, 0.0, 2.0 * np.pi)
    # Evaluation point sample of a Spherical Harmonic basis function.
    var_type = theta.dtype
    sign_m = tf.math.sign(m)
    m = tf.abs(m)
    zeros = tf.zeros_like(m)
    result_m_zero = _spherical_harmonics_normalization(
        l, zeros, var_type) * evaluate_legendre_polynomial(
            l, zeros, tf.cos(theta))
    result_branch = _evaluate_spherical_harmonics_branch(
        l, m, theta, phi, sign_m, var_type)
    return tf.where(tf.equal(m, zeros), result_m_zero, result_branch)


def rotate_zonal_harmonics(zonal_coeffs, theta, phi, name=None):
  """Rotates zonal harmonics.

  Args:
    zonal_coeffs: N-D tensor of any storing zonal harmonics coefficients.
    theta: N-D tensor of any shape storing the polar angles.
    phi: N-D tensor of any shape storing the azimuthal angles.
    name: A name for this op. Defaults to
      'spherical_harmonics_rotate_zonal_harmonics'.

  Returns:
    N-D tensor of shape `[?, ..., ?]` storing coefficients of the rotated
    harmonics.

  Raises:
    ValueError: if `theta` or `phi` have different shapes.
  """
  with tf.compat.v1.name_scope(name,
                               "spherical_harmonics_rotate_zonal_harmonics",
                               [zonal_coeffs, theta, phi]):
    zonal_coeffs = tf.convert_to_tensor(value=zonal_coeffs)
    theta = tf.convert_to_tensor(value=theta)
    phi = tf.convert_to_tensor(value=phi)
    if theta.shape.as_list() != phi.shape.as_list():
      raise ValueError("'theta' and 'phi' have different shapes.")
    tiled_zonal_coeffs = tile_zonal_coefficients(zonal_coeffs)
    max_band = zonal_coeffs.shape.as_list()[-1]
    l, m = generate_l_m_permutations(max_band - 1)
    broadcast_shape = theta.shape.as_list()[:-1] + l.shape.as_list()
    l_broadcasted = tf.broadcast_to(l, broadcast_shape)
    m_broadcasted = tf.broadcast_to(m, broadcast_shape)
    n_star = tf.sqrt(4.0 * np.pi / (2.0 * tf.cast(l, dtype=theta.dtype) + 1.0))
    return n_star * tiled_zonal_coeffs * evaluate_spherical_harmonics(
        l_broadcasted, m_broadcasted, theta, phi)


def tile_zonal_coefficients(coefficients, name=None):
  """Tiles zonal coefficients.

  Zonal Harmonics only contains the harmonics where m=0. This function returns
  these coefficients for -l <= m <= l, where l is the rank of `coefficients`.

  Args:
    coefficients: 1-d tensor.
    name: A name for this op. Defaults to
      'spherical_harmonics_tile_zonal_coefficients'.

  Return: A 1-d tensor.

  Raises:
    ValueError: if the shape of `coefficients` is not supported.
  """
  with tf.compat.v1.name_scope(
      name, "spherical_harmonics_tile_zonal_coefficients", [coefficients]):
    coefficients = tf.convert_to_tensor(value=coefficients)
    if len(coefficients.shape.as_list()) != 1:
      raise ValueError("'coefficients' must have 1 dimension.")
    return tf.concat([
        pair[1] * tf.ones(shape=(2 * pair[0] + 1,), dtype=coefficients.dtype)
        for pair in enumerate(tf.unstack(coefficients, axis=0))
    ],
                     axis=0)


# API contains all public functions.
__all__ = [
    obj_name for obj_name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and not obj_name.startswith("_")
]
