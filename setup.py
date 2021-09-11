#!/usr/bin/env python

"""
Copyright (c) 2021
Argo AI, LLC, All Rights Reserved.

Notice: All information contained herein is, and remains the property
of Argo AI. The intellectual and technical concepts contained herein
are proprietary to Argo AI, LLC and may be covered by U.S. and Foreign
Patents, patents in process, and are protected by trade secret or
copyright law. This work is licensed under a CC BY-NC-SA 4.0 
International License.

Originating Authors: John Lambert
"""

"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import platform
import sys
from codecs import open  # To use a consistent encoding
from os import path

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


if platform.system() == "Windows":
    print("mseg-semantic currently does not support Windows, please use Linux/Mac OS")
    sys.exit(1)

setup(
    name="tbv",
    version="0.1.0",
    description="",
    long_description=long_description,
    url="",
    author="Argo AI",
    author_email="johnlambert@gatech.edu",
    license="CC BY-NC-SA 4.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="computer-vision dataset-tools",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    python_requires=">= 3.6",
)
