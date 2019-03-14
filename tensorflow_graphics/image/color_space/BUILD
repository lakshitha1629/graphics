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
# Color space utils for tf-graphics.

# google internal)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "color_space",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":srgb",
    ],
)

py_library(
    name = "srgb",
    srcs = ["srgb.py"],
    srcs_version = "PY2AND3",
    deps = [
        # google internal,
        "//tensorflow_graphics/util:asserts",
    ],
)

py_test(
    name = "srgb_test",
    srcs = ["tests/srgb_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":srgb",
        # package dep

        # google internal,
        "//tensorflow_graphics/util:test_case",
    ],
)
