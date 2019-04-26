<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.interpolation.bspline.interpolate_with_weights" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.interpolation.bspline.interpolate_with_weights

Interpolates knots using knot weights.

``` python
tfg.math.interpolation.bspline.interpolate_with_weights(
    knots,
    weights,
    name=None
)
```



Defined in [`math/interpolation/bspline.py`](https://cs.corp.google.com/#piper///depot/google3/third_party/py/tensorflow_graphics/math/interpolation/bspline.py).

<!-- Placeholder for "Used in" -->

Note:
  In the following, A1 to An, and B1 to Bk are optional batch dimensions.

#### Args:

* <b>`knots`</b>: A tensor with shape `[B1, ..., Bk, num_knots]` containing knot
    values.
* <b>`weights`</b>: A tensor with shape `[A1, ..., An, num_splines, num_knots]`
    containing dense weights for knots.
* <b>`name`</b>: A name for this op. Defaults to "bsplines_interpolate_with_weights".


#### Returns:

A tensor with shape `[A1, ..., An, B1, ..., Bk]`, which is the result of
spline interpolation.


#### Raises:

* <b>`ValueError`</b>: If the last dimension of knots and weights is not equal.