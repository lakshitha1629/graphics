<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.representation.point.distance_to_ray" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.representation.point.distance_to_ray

Computes the distance from a M-d point to a M-d ray.

``` python
tfg.geometry.representation.point.distance_to_ray(
    point,
    origin,
    direction,
    name=None
)
```



Defined in [`geometry/representation/point.py`](https://cs.corp.google.com/#piper///depot/google3/third_party/py/tensorflow_graphics/geometry/representation/point.py).

<!-- Placeholder for "Used in" -->

Note:
  In the following, A1 to An are optional batch dimensions.

#### Args:

* <b>`point`</b>: A tensor of shape `[A1, ..., An, M]`.
* <b>`origin`</b>: A tensor of shape `[A1, ..., An, M]`.
* <b>`direction`</b>: A tensor of shape `[A1, ..., An, M]`. The last dimension must be
    normalized.
* <b>`name`</b>: A name for this op. Defaults to "point_distance_to_ray".


#### Returns:

A tensor of shape `[A1, ..., An, 1]` containing the distance from each point
to the corresponding ray.


#### Raises:

* <b>`ValueError`</b>: If the shape of `point`, `origin`, or 'direction' is not
  supported.