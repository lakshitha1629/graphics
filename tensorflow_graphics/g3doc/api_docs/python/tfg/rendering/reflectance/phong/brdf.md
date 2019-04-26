<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.reflectance.phong.brdf" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.reflectance.phong.brdf

Evaluates the specular brdf of the Phong model.

``` python
tfg.rendering.reflectance.phong.brdf(
    direction_incoming_light,
    direction_outgoing_light,
    surface_normal,
    shininess,
    albedo,
    name=None
)
```



Defined in [`rendering/reflectance/phong.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/reflectance/phong.py).

<!-- Placeholder for "Used in" -->

Note:
  This function returns a modified specular Phong model that ensures energy
  conservation.

Note:
  In the following, A1 to An are optional batch dimensions.

#### Args:

* <b>`direction_incoming_light`</b>: A tensor of shape `[A1, ..., An, 3]`, where the
    last dimension represents a normalized incoming light vector.
* <b>`direction_outgoing_light`</b>: A tensor of shape `[A1, ..., An, 3]`, where the
    last dimension represents a normalized outgoing light vector.
* <b>`surface_normal`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last
    dimension represents a normalized surface normal.
* <b>`shininess`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
    represents a shininess coefficient.
* <b>`albedo`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
    represents albedo with values in [0,1].
* <b>`name`</b>: A name for this op. Defaults to "phong_brdf".


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
  the amount of light reflected in the outgoing light direction.


#### Raises:

* <b>`ValueError`</b>: if the shape of `direction_incoming_light`,
  `direction_outgoing_light`, `surface_normal`, `shininess` or `albedo` is not
  supported.
* <b>`InvalidArgumentError`</b>: if at least an element of `albedo` is outside of
  [0,1].