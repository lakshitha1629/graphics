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
# Utility functionalities for tf-graphics.

# google internal)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "test_case",
    srcs = ["test_case.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tfg_flags",
        # package dep,
        # package dep

        # google internal,
    ],
)

py_library(
    name = "tfg_flags",
    srcs = ["tfg_flags.py"],
    srcs_version = "PY2AND3",
    deps = [
        # package dep,
    ],
)

py_library(
    name = "asserts",
    srcs = ["asserts.py"],
    srcs_version = "PY2AND3",
    deps = [
        # package dep,

        # google internal,
        "//tensorflow_graphics/util:tfg_flags",
    ],
)

py_library(
    name = "math",
    srcs = ["math.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":asserts",

        # google internal,
        "//tensorflow_graphics/util:safe_ops",
    ],
)

py_library(
    name = "safe_ops",
    srcs = ["safe_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":asserts",
        ":tfg_flags",
        # package dep,

        # google internal,
    ],
)

py_library(
    name = "shape",
    srcs = ["shape.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "util",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":asserts",
        ":math",
        ":safe_ops",
        ":shape",
        ":test_case",
        ":tfg_flags",
    ],
)

py_test(
    name = "asserts_test",
    srcs = ["tests/asserts_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":asserts",
        # package dep
        # package dep

        # google internal,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "math_test",
    srcs = ["tests/math_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":math",
        # package dep

        # google internal,
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "safe_ops_test",
    srcs = ["tests/safe_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":asserts",
        ":safe_ops",
        # package dep

        # google internal,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "shape_test",
    srcs = ["tests/shape_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":shape",
        # package dep
        # google internal,
        "//tensorflow_graphics/util:test_case",
    ],
)
