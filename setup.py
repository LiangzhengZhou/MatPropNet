"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import find_packages, setup

setup(
    name="matpropnet",
    version="0.1.0",
    description="General materials property prediction framework with decoupled graph backbones",
    url="https://github.com/bt403/ocp-gemnet-gnn",
    packages=find_packages(),
    include_package_data=True,
)
