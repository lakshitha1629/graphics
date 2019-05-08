# TensorFlow Graphics

## The repository is currently in development. The library will be officially released the 9th of May, including the pip package.

[![PyPI project status](https://img.shields.io/pypi/status/tensorflow-graphics.svg)](https://pypi.org/project/tensorflow-graphics/)
[![Travis build status](https://img.shields.io/travis/tensorflow/graphics.svg)](https://travis-ci.org/tensorflow/graphics)
[![Code coverage](https://img.shields.io/coveralls/github/tensorflow/graphics.svg)](https://coveralls.io/github/tensorflow/graphics)
[![Supported Python version](https://img.shields.io/pypi/pyversions/tensorflow-graphics.svg)](https://pypi.org/project/tensorflow-graphics/)
[![PyPI release version](https://img.shields.io/pypi/v/tensorflow-graphics.svg)](https://pypi.org/project/tensorflow-graphics/)

The last few years have seen a rise in novel differentiable graphics layers
which can be inserted in neural network architectures. From spatial transformers
to differentiable graphics renderers, these new layers leverage the knowledge
acquired over years of computer vision and graphics research to build new and
more efficient network architectures. Explicitly modeling geometric priors and
constraints into neural networks opens up the door to architectures that can be
trained robustly, efficiently, and more importantly, in a self-supervised
fashion.

## Overview

At a high level, a computer graphics pipeline requires 3D objects and their
absolute positioning in the scene, a description of the material they are made
of, lights and a camera. This scene description is then interpreted by a
renderer to generate a synthetic rendering.

<div align="center">
  <img border="0"  src="https://storage.googleapis.com/tensorflow-graphics/git/readme/graphics.jpg" width="600">
</div>

In comparison, a computer vision system would start from an image and try to
infer the parameters of the scene. This allows the prediction of which objects
are in the scene, what materials they are made and their three dimensional
\position and orientation.


<div align="center">
  <img border="0"  src="https://storage.googleapis.com/tensorflow-graphics/git/readme/cv.jpg" width="600">
</div>


Training machine learning systems capable of solving these complex 3D vision
tasks most often requires large quantities of data. As labelling data is a
costly and complex process, it is important to have mechanisms to design
machine learning models that can comprehend the three dimensional world while
being trained without much supervision. Combining computer vision and computer
graphics techniques provides a unique opportunity to leverage the vast amounts
of readily available unlabelled data. This can, for instance, be achieved using
analysis by synthesis


<div align="center">
  <img border="0"  src="https://storage.googleapis.com/tensorflow-graphics/git/readme/cv_graphics.jpg" width="600">
</div>

where the vision system extracts the scene parameters and the graphics system
renders back an image based on them. If the rendering matches the original
image, the vision system has accurately extracted the scene parameters. In this
setup, computer vision and computer graphics go hand in hand, forming a single
machine learning system similar to an autoencoder, which can be trained in a
self-supervised manner.

Tensorflow Graphics is being developed to help tackle these types of challenges
and to do so, provides a set of differentiable graphics layers (e.g. cameras and
reflectance models) and 3D functionalities (e.g. mesh convolution, 3D
tensorboard) that can be used to train and debug your machine learning models of
choice.

## Installing TensorFlow Graphics

See the [install](tensorflow_graphics/g3doc/install.md) documentation for
instructions on how to install TensorFlow Graphics.

## API Documentation
You can find the API documentation [here](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/g3doc/api_docs/python/tfg.md).

## Compatibility
TensorFlow Graphics is fully compatible with the latest stable release of
TensorFlow, tf-nightly, and tf-nightly-2.0-preview. All the functions are
compatible with graph and eager execution.

## Colab tutorials
To help you get started with some of the functionalities provided by TF
Graphics, some Colab notebooks are available below and roughly ordered by
difficulty. These Colabs touch upon a large range of topics including, object
pose estimation, interpolation, object materials, lighting, and non-rigid
surface deformation.

NOTE: the tutorials are maintained carefully. However, they are not considered
part of the API and they can change at any time without warning. It is not
advised to write code that takes dependency on them.

### Beginner

<div align="center">
  <table>
    <tr>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/6dof_alignment.ipynb">Object pose estimation</a></th>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/intrinsics_optimization.ipynb">Camera intrisic optimization</a></th>
    </tr>
    <tr>
      <td align="center">
        <a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/6dof_alignment.ipynb"><img border="0"  src="https://storage.googleapis.com/tensorflow-graphics/notebooks/6dof_pose/thumbnail.jpg" width="200" height="200">
        </a>
      </td>
      <td align="center">
              <a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/intrinsics_optimization.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/intrinsics/intrinsics_thumbnail.png" width="200" height="200">
        </a>
      </td>
    </tr>
  </table>
</div>

### Intermediate

<div align="center">
  <table>
    <tr>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/interpolation.ipynb">B-spline and slerp interpolation</a></th>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/reflectance.ipynb">Reflectance</a></th>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/non_rigid_deformation.ipynb">Non-rigid surface deformation</a></th>
    </tr>
    <tr>
      <td align="center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/interpolation.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/interpolation/thumbnail.png" width="200" height="200"> </td>
      <td align="center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/reflectance.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/reflectance/thumbnail.png" width="200" height="200"></td>
      <td align="center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/non_rigid_deformation.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/non_rigid_deformation/thumbnail.jpg" width="200" height="200">
      </a></td>
    </tr>
  </table>
</div>


### Advanced
<div align="center">
  <table>
    <tr>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/spherical_harmonics_approximation.ipynb">Spherical harmonics rendering</a></th>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/spherical_harmonics_optimization.ipynb">Environment map optimization</a></th>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/mesh_segmentation_demo.ipynb">Semantic mesh segmentation</a></th>
    </tr>
    <tr>
      <td align="center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/spherical_harmonics_approximation.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/sh_rendering/thumbnail.png" width="200" height="200">
      </a></td>
      <td align="center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/spherical_harmonics_optimization.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/environment_lighting/thumbnail.png" width="200" height="200">
      </a></td>
      <td align="center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/mesh_segmentation_demo.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/mesh_segmentation/thumbnail.jpg" width="200" height="200">
      </a></td>
    </tr>
  </table>
</div>

## Coming next...
Among many things, we are hoping to release resamplers, additional 3D
convolution and pooling operators, and a differentiable rasterizer!

Follow us on [Twitter](https://twitter.com/_TFGraphics_) to hear about the
latest updates!

## Additional Information
You may use this software under the [Apache 2.0 License](https://github.com/tensorflow/graphics/blob/master/LICENSE).

## Community
As part of TensorFlow, we're committed to fostering an open and welcoming
environment.

* [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow): Ask
  or answer technical questions.
* [GitHub](https://github.com/tensorflow/graphics/issues): Report bugs or
  make feature requests.
* [TensorFlow Blog](https://medium.com/tensorflow): Stay up to date on content
  from the TensorFlow team and best articles from the community.
* [Youtube Channel](http://youtube.com/tensorflow/): Follow TensorFlow shows.

## References
If you use TensorFlow Graphics in your research, please reference it as:

    @inproceedings{TensorflowGraphicsIO2019,
       author = {Valentin, Julien and Keskin, Cem and Pidlypenskyi, Pavel and Makadia, Ameesh and Sud, Avneesh and Bouaziz, Sofien},
       title = {TensorFlow Graphics: Computer Graphics Meets Deep Learning},
       year = {2019}
    }

### Contributors - in alphabetical order
- Sofien Bouaziz
- Jay Busch
- Forrester Cole
- Ambrus Csaszar
- Boyang Deng
- Ariel Gordon
- Cem Keskin
- Ameesh Makadia
- Rohit Pandey
- Pavel Pidlypenskyi
- Avneesh Sud
- Anastasia Tkach
- Julien Valentin
- He Wang
- Yinda Zhang




