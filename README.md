# TensorFlow Graphics

## The repository is currently in development. The library will be officially released the 7th of May, including the pip package.

[![PyPI project status](https://img.shields.io/pypi/status/tensorflow-graphics.svg)]()
[![Travis build status](https://img.shields.io/travis/tensorflow/graphics.svg)](https://travis-ci.org/tensorflow/graphics)
[![Code coverage](https://img.shields.io/coveralls/github/tensorflow/graphics.svg)](https://coveralls.io/github/tensorflow/graphics)
[![Supported Python version](https://img.shields.io/pypi/pyversions/tensorflow-graphics.svg)]()
[![PyPI release version](https://img.shields.io/pypi/v/tensorflow-graphics.svg)](https://pypi.org/project/tensorflow-graphics/)

At a high level, a computer graphics pipeline needs 3D objects, the value of
their rotation and translation in the scene, a description of the material they
are made of, lights, and a camera. These parameters are then interpreted by a
renderer to generate an image of the scene.

<div align="center">
  <img border="0"  src="https://storage.googleapis.com/tensorflow-graphics/git/readme/graphics.jpg" width="600">
</div>

In comparison, a computer vision system would start from an image and try to
infer the paramaters that allowed to create this scene, allowing to answer
questions like 'which objects are in the scene?', 'what is the 3D position of
these objects?', 'what materials are objects made of?'.

<div align="center">
  <img border="0"  src="https://storage.googleapis.com/tensorflow-graphics/git/readme/cv.jpg" width="600">
</div>


Training machine learning systems capable of solving these complex 3D vision
tasks requires large quantities of data. Labelling data being a costly and
complex process, it is important to have mechanisms allowing to design machine
learning models that can reason about the three dimensional world while being
trained without much supervision. Combining computer vision and computer
graphics techniques provides a unique opportunity to leverage the vast amount
of readily available unlabelled data. This can for instance be achieved using
analysis by synthesis.

<div align="center">
  <img border="0"  src="https://storage.googleapis.com/tensorflow-graphics/git/readme/cv_graphics.jpg" width="600">
</div>


Tensorflow Graphics is being developed to help tackle these types of challenges
and to do so, provides a set of differentiable graphics layers (e.g. cameras and
reflectance models) and 3D functionalities (e.g. mesh convolution, 3D
tensorboard) that can be used to train and debug your machine learning models of
choice.

## Installing TensorFlow Graphics

### Stable builds
TensorFlow Graphics depends on [TensorFlow](https://www.tensorflow.org/install)
1.13.1 or above. Nightly builds of TensorFlow that tf-nightly and
tf-nightly-2.0-preview are also supported.

To install the latest CPU version from
[PyPI](https://pypi.org/project/tensorflow-graphics/), run the following:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --upgrade tensorflow-graphics
```

and to install the latest GPU version, run:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --upgrade tensorflow-graphics-gpu
```

For additional installation help, guidance installing prerequisites,
and (optionally) setting up virtual environments, see the [TensorFlow
installation guide](https://www.tensorflow.org/install).

### Installing from source - Linux
You can also install from source by executing the following commands:

```shell
git clone https://github.com/tensorflow/graphics.git
sh build_pip_pkg.sh
pip install --upgrade dist/*.whl
```

### Installing optional packages - Linux
To use the TensorFlow Graphics EXR data loader,
OpenEXR needs to be installed. This can be done by running the following
commands:

```
sudo apt-get install libopenexr-dev
pip install --upgrade OpenEXR
```

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

## API Documentation
You can find the API documentation [here](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/g3doc/api_docs/python/tfg.md).

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
      <th style="text-align:center"><a href="https://colab.sandbox.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/6dof_alignment.ipynb">Object pose estimation</a></th>
      <th style="text-align:center"><a href="https://colab.sandbox.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/intrinsics_optimization.ipynb">Camera intrisic optimization</a></th>
    </tr>
    <tr>
      <td style="text-align:center">
        <a href="https://colab.sandbox.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/6dof_alignment.ipynb"><img border="0"  src="https://storage.googleapis.com/tensorflow-graphics/notebooks/6dof_pose/thumbnail.jpg" width="200" height="200">
        </a>
      </td>
      <td style="text-align:center">
              <a href="https://colab.sandbox.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/intrinsics_optimization.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/intrinsics/intrinsics_thumbnail.png" width="200" height="200">
        </a>
      </td>   
    </tr>
  </table>
</div>

### Intermediate

<div align="center">
  <table>
    <tr>
      <th style="text-align:center"><a href="https://colab.sandbox.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/interpolation.ipynb">B-spline and slerp interpolation</a></th>
      <th style="text-align:center"><a href="https://colab.sandbox.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/reflectance.ipynb">Reflectance</a></th>
    </tr>
    <tr>
      <td><a href="https://colab.sandbox.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/interpolation.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/interpolation/thumbnail.png" width="200" height="200"> </td>
      <td><a href="https://colab.sandbox.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/reflectance.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/reflectance/thumbnail.png" width="200" height="200"></td>
    </tr>
  </table>
</div>


### Advanced
<div align="center">
  <table>
    <tr>
      <th style="text-align:center"><a href="https://colab.sandbox.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/spherical_harmonics_approximation.ipynb">Spherical harmonics rendering</a></th>
      <th style="text-align:center"><a href="https://colab.sandbox.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/spherical_harmonics_optimization.ipynb">Environment map optimization</a></th>
      <th style="text-align:center"><a href="https://colab.sandbox.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/non_rigid_deformation.ipynb">Non-rigid surface deformation</a></th>
    </tr>
    <tr>
      <td style="text-align:center"><a href="https://colab.sandbox.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/spherical_harmonics_approximation.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/sh_rendering/thumbnail.png" width="200" height="200">
      </a></td>
      <td style="text-align:center"><a href="https://colab.sandbox.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/spherical_harmonics_optimization.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/environment_lighting/thumbnail.png" width="200" height="200">
      </a></td>
      <td style="text-align:center"><a href="https://colab.sandbox.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/non_rigid_deformation.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/non_rigid_deformation/thumbnail.jpg" width="200" height="200">
      </a></td>
    </tr>
  </table>
</div>

## Additional Information
You may use this software under the [Apache 2.0 License](https://github.com/tensorflow/graphics/blob/master/LICENSE).

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




