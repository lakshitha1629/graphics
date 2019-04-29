# Tensorflow graphics library.

## The repository is currently in development. The library will be officially released the 7th of May, including the pip package.

TensorFlow Graphics is a library of graphics related Tensorflow ops. As part of 
the TensorFlow ecosystem, these graphics ops can also be used to build machine 
learning models and are for the most part differentiable.


During the last few years we have seen a rise in the creation of novel
differentiable graphics layers that can be inserted in standard neural network
architectures. From spatial transformers to differentiable graphics renderers,
these new layers allow to use the knowledge acquired in years of computer vision
and graphics research for the design of efficient network architectures.
Explicitly modeling geometric priors and constraints into a neural networks can 
help learning invariance to 3D geometric transformations, and also opens up the 
door to architectures that can be trained in a self-supervised or even fully
unsupervised fashion. TensorFlow Graphics aims at bringing some of these 
functionalities into TensorFlow to stimulate research in this field by making 
useful graphics functions widely accessible to the community.


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
installed version of TensorFlow. You can also directly specify which TensorFlow
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
pip install --upgrade OpenEXR
```

## API Documentation
You can find our API documentation [here](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/g3doc/api_docs/python/tfg.md).

## Tensorboard Plugin
Coming soon...

# Notebooks
Coming soon...

# Additional Information
You may use this software under the Apache 2.0 License. See [LICENSE](https://github.com/tensorflow/graphics/blob/master/LICENSE).


