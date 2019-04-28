<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.convolution.utils" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfg.geometry.convolution.utils

This module implements various sparse data utilities for graphs and meshes.

Defined in
[`geometry/convolution/utils.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/convolution/utils.py).

<!-- Placeholder for "Used in" -->

## Functions

[`check_valid_graph_convolution_input(...)`](../../../tfg/geometry/convolution/utils/check_valid_graph_convolution_input.md):
Checks that the inputs are valid for graph convolution ops.

[`convert_to_block_diag_2d(...)`](../../../tfg/geometry/convolution/utils/convert_to_block_diag_2d.md):
Convert a batch of 2d SparseTensors to a 2d block diagonal SparseTensor.

[`flatten_batch_to_2d(...)`](../../../tfg/geometry/convolution/utils/flatten_batch_to_2d.md):
Reshape a batch of 2d Tensors by flattening across the batch dimensions.

[`partition_sums_2d(...)`](../../../tfg/geometry/convolution/utils/partition_sums_2d.md):
Sum over subsets of rows in a 2-D tensor.
