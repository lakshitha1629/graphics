<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.interpolation.bspline.knot_weights" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.interpolation.bspline.knot_weights

Function that converts cardinal B-spline positions to knot weights.

``` python
tfg.math.interpolation.bspline.knot_weights(
    positions,
    num_knots,
    degree,
    cyclical,
    sparse_mode=False,
    name=None
)
```



Defined in [`math/interpolation/bspline.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/interpolation/bspline.py).

<!-- Placeholder for "Used in" -->


Note:
  In the following, A1 to An are optional batch dimensions.

#### Args:

* <b>`positions`</b>: A tensor of shape `[A1, ... An]` of positions. These values must
    be in range [0,`num_knots`-`degree`) for non-cyclical splines and [0,
    `num_knots`) for cyclical splines. Dimensions can be `None`.
* <b>`num_knots`</b>: A strictly positive `int` describing the number of knots in the
    spline.
* <b>`degree`</b>: An `int` describing the degree of the spline, which must be smaller
    than `num_knots`.
* <b>`cyclical`</b>: A `bool` describing whether the spline is cyclical.
* <b>`sparse_mode`</b>: A `bool` describing whether to return a result only for the
    knots with nonzero weights. If set to True, the function returns the
    weights of only the `degree` + 1 knots that are non-zero, as well as the
    indices of the knots.
* <b>`name`</b>: A name for this op. Defaults to "bsplines_knot_weights".


#### Returns:

A tensor with dense weights for each control point, with the shape
`[A1, ... An, num_knots]` if `sparse_mode` is False.
Otherwise, returns a tensor of shape `[A1, ... An, degree + 1]` that
contains the non-zero weights, and a tensor with the indices of the knots,
with the type tf.int32.


#### Raises:

* <b>`ValueError`</b>: If degree is larger than num_knots - 1.
* <b>`NotImplementedError`</b>: If degree > 4 or < 0
* <b>`InvalidArgumentError`</b>: If positions are not in the right range.