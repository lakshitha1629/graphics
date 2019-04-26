<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.reflectance.lambertian.brdf" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.reflectance.lambertian.brdf

Evaluates the brdf of a Lambertian surface.

``` python
tfg.rendering.reflectance.lambertian.brdf(
    albedo,
    name=None
)
```



Defined in [`rendering/reflectance/lambertian.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/reflectance/lambertian.py).

<!-- Placeholder for "Used in" -->

Note:
  In the following, A1 to An are optional batch dimensions.

#### Args:

* <b>`albedo`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
    represents albedo with values in [0,1].
* <b>`name`</b>: A name for this op. Defaults to "lambertian_brdf".


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
  the amount of reflected light in any outgoing direction.


#### Raises:

* <b>`ValueError`</b>: if the shape of `albedo` is not supported.
* <b>`InvalidArgumentError`</b>: if at least an element of `albedo` is outside of
  [0,1].