<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.representation.mesh.convolution.utils.partition_sums_2d" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.representation.mesh.convolution.utils.partition_sums_2d

Sum over subsets of rows in a 2-D tensor.

``` python
tfg.geometry.representation.mesh.convolution.utils.partition_sums_2d(
    data,
    group_ids,
    row_weights=None,
    name=None
)
```



Defined in [`geometry/representation/mesh/convolution/utils.py`](https://cs.corp.google.com/#piper///depot/google3/third_party/py/tensorflow_graphics/geometry/representation/mesh/convolution/utils.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`data`</b>: 2-D tensor with shape `[D1, D2]`.
* <b>`group_ids`</b>: 1-D Int tensor with shape `[D1]`.
* <b>`row_weights`</b>: 1-D tensor with shape `[D1]`. Can be None.
* <b>`name`</b>: A name for this op. Defaults to `utils_partition_sums_2d`.


#### Returns:

A 2-D tensor with shape `[max(group_ids) + 1, D2]` where
  `output[i, :] = sum(data[j, :] * weight[j] * 1(group_ids[j] == i)),
  1(.) is the indicator function.


#### Raises:

* <b>`ValueError`</b>: if the inputs have invalid dimensions or types.