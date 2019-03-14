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
# Geometry utils for tf-graphics.

# google internal)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "point",
    srcs = ["point.py"],
    srcs_version = "PY2AND3",
    deps = [
        # google internal,
        "//tensorflow_graphics/util:asserts",
    ],
)

py_library(
    name = "grid",
    srcs = ["grid.py"],
    srcs_version = "PY2AND3",
    deps = [
        # google internal,
    ],
)

py_library(
    name = "triangle",
    srcs = ["triangle.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":vector",
        # google internal,
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "vector",
    srcs = ["vector.py"],
    srcs_version = "PY2AND3",
    deps = [
        # google internal,
        "//tensorflow_graphics/util:asserts",
    ],
)

py_library(
    name = "geometry",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":grid",
        ":point",
        ":ray",
        ":triangle",
        ":vector",
    ],
)

py_library(
    name = "ray",
    srcs = ["ray.py"],
    srcs_version = "PY2AND3",
    deps = [
        # google internal,
    ],
)

py_test(
    name = "point_test",
    srcs = ["tests/point_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":point",
        # package dep
        # package dep

        # google internal,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "triangle_test",
    srcs = ["tests/triangle_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":triangle",
        ":vector",
        # package dep

        # google internal,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "vector_test",
    srcs = ["tests/vector_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":vector",
        # package dep
        # package dep

        # google internal,
        "//tensorflow_graphics/transformation:test_data",
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "grid_test",
    srcs = ["tests/grid_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":grid",
        # package dep

        # google internal,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "ray_test",
    srcs = ["tests/ray_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ray",
        # package dep,
        # package dep

        # google internal,
        "//tensorflow_graphics/util:test_case",
    ],
)
