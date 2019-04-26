<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.spherical_harmonics.integration_product" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.spherical_harmonics.integration_product

Computes the integral of f.g over the sphere.

``` python
tfg.math.spherical_harmonics.integration_product(
    f,
    g,
    keepdims=True,
    name=None
)
```



Defined in [`math/spherical_harmonics.py`](https://github.com/tensorflow/agents/tree/master/tensorflow_graphics/math/spherical_harmonics.py).

<!-- Placeholder for "Used in" -->

Note:
  In the following, A1 to An are optional batch dimensions.

#### Args:

* <b>`f`</b>: A tensor of shape `[A1, ..., An, C]`, where the last dimension represents
    spherical harmonics coefficients.
* <b>`g`</b>: A tensor of shape `[A1, ..., An, C]`, where the last dimension represents
    spherical harmonics coefficients.
* <b>`keepdims`</b>: If true, retains reduced dimensions with length 1.
* <b>`name`</b>: A name for this op. Defaults to "spherical_harmonics_convolution".


#### Returns:

A tensor of shape `[A1, ..., An]` containing scalar values resulting from
integrating the product of the spherical harmonics `f` and `g`.


#### Raises:

* <b>`ValueError`</b>: if the last dimension of `f` is different from the last
  dimension of `g`.