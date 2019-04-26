# Tensorflow graphics library.

TensorFlow Graphics is a library of graphics related Tensorflow ops. As part of 
the TensorFlow ecosystem, these graphics ops can also be used to build machine 
learning models and are for the most part differentiable.


[![PyPI project status](https://img.shields.io/pypi/status/tensorflow-graphics.svg)]()
[![Travis build status](https://img.shields.io/travis/tensorflow/graphics.svg)](https://travis-ci.org/tensorflow/graphics)
[![Code coverage](https://img.shields.io/coveralls/github/tensorflow/graphics.svg)](https://coveralls.io/github/tensorflow/graphics)
[![Supported Python version](https://img.shields.io/pypi/pyversions/tensorflow-graphics.svg)]()
[![PyPI release version](https://img.shields.io/pypi/v/tensorflow-graphics.svg)](https://pypi.org/project/tensorflow-graphics/)
 
* [👍 **Get Started**](#get-started)
* [📓 **Notebooks**](#notebooks) 
* [🔧 **Additional Information**](#additional-information)
* [💬 **Community**](#community)

# Get Started

## Install TensorFlow Graphics
To install the latest version of TensorFlow Graphics, run the following command:

```
pip install --upgrade tensorflow-graphics
```

TensorFlow Graphics depends on a recent stable release of TensorFlow. Since
TensorFlow is not included as a dependency of the TensorFlow Graphics package,
you must explicitly install the TensorFlow package.
This can be done manually by using 

```
pip install --upgrade tensorflow
```

for the CPU version or

```
pip install --upgrade tensorflow-gpu
```

for the GPU version. While in practice, we recommend you to use your already
installed version of TensorGlow. You can also directly specify which TensorFlow
version TensorFlow Graphics should install using

```
pip install --upgrade tensorflow-graphics[tf]
```
```
pip install --upgrade tensorflow-graphics[tf-gpu]
```

[**Optional**] To use the TensorFlow Graphics EXR data loader,
OpenEXR needs to be installed. This can be done by running the following
commands:

```
sudo apt-get -y install libopenexr-dev
pip install OpenEXR
```

## API Documentation
You can find our API documentation [here](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/g3doc/api_docs/python/tfg.md).

## Tensorboard Plugin
Coming soon...

# Notebooks

<a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/6dof_alignment.ipynb">
<img src="https://storage.googleapis.com/tensorflow-graphics/notebooks/6dof_pose/task.jpg" width="500" alt=""></img>
</a>


# Additional Information
You may use this software under the Apache 2.0 License. See [LICENSE](https://github.com/tensorflow/graphics/blob/master/LICENSE).

# Community
We are in `#tf-graphics` on the tf-graphics slack channel ([join link](https://join.slack.com/t/tf-graphics/shared_invite/enQtNjE5NTQ1NTg5ODYwLWU3MzQ2YTEzZTdkN2RhMzgwMWM5MzdhMDRmMWRlM2E3MzhhMjAyYmZhNWM2OWQ2Y2ExNGUzNTE1Y2Y4MjZhOWU)). We would love to hear from you!

