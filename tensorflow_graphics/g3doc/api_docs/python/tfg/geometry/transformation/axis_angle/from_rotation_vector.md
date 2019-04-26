<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.axis_angle.from_rotation_vector" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.axis_angle.from_rotation_vector

Converts a rotation vector to an axis-angle representation.

``` python
tfg.geometry.transformation.axis_angle.from_rotation_vector(
    rotation_vector,
    name=None
)
```



Defined in [`geometry/transformation/axis_angle.py`](https://github.com/tensorflow/agents/tree/master/tensorflow_graphics/geometry/transformation/axis_angle.py).

<!-- Placeholder for "Used in" -->

A rotation vector is a vector $$r \in \mathbb{R}^3$$ where
$$\frac{r}{\|r\|_2} = \mathbf{a}$$ is a unit vector indicating the axis of
rotation and $$\|r\|_2 = \theta$$ is the angle.

Note:
  In the following, A1 to An are optional batch dimensions.

#### Args:

* <b>`rotation_vector`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last
    dimension represents a rotation vector.
* <b>`name`</b>: A name for this op that defaults to "axis_angle_from_rotation_vector".


#### Returns:

A tuple of two tensors, respectively of shape `[A1, ..., An, 3]` and
`[A1, ..., An, 1]`, where the first tensor represents the axis, and the
second represents the angle. The resulting axis is a normalized vector.


#### Raises:

* <b>`ValueError`</b>: If the shape of `rotation_vector` is not supported.