#Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Tensorflow-Graphics, a library for differentiable computer graphics and geometry.

# google internal
# google internal

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

py_library(
    name = "tensorflow_graphics",
    srcs = [
        "__init__.py",
        "version.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_graphics/camera",
        "//tensorflow_graphics/deformation_energies",
        "//tensorflow_graphics/geometry",
        "//tensorflow_graphics/image",
        "//tensorflow_graphics/interpolation",
        "//tensorflow_graphics/lighting",
        "//tensorflow_graphics/reflectance",
        "//tensorflow_graphics/transformation",
        "//tensorflow_graphics/util",
    ],
)
