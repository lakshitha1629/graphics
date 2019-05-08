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
"""Helper methods for downloading resources for Colab demos.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import six.moves.urllib.parse as urlparse
import six.moves.urllib.request as urlrequest


def _create_dir(local_dir):
  """Creates a directory recursively if it does not exist."""
  if not os.path.exists(local_dir):
    os.makedirs(local_dir)


def _download_file(remote_url, local_file):
  """Downloads file from remote url to local file if not already downloaded.

  Args:
    remote_url: A valid URL of a remote file.
    local_file: Full path filename where the remote file should be saved.
  """
  if not os.path.isfile(local_file):
    print('Downloading {0} to {1}'.format(remote_url, local_file))
    filedata = urlrequest.urlopen(remote_url)
    datatowrite = filedata.read()
    with open(local_file, 'wb') as writer:
      writer.write(datatowrite)


def download_files(remote_url_base, data_files, local_folder):
  """Downloads a set of files from remote URL to local folder.

  Args:
    remote_url_base: A valid base URL prefix to data_files.
    data_files: A list of files present at remote_url_base.
    local_folder: Local folder where the data files should be saved.
  """
  _create_dir(local_folder)
  for data_file in data_files:
    remote_url = urlparse.urljoin(remote_url_base, data_file)
    local_data_file = os.path.join(local_folder, data_file)
    _download_file(remote_url, local_data_file)
