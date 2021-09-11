#!/bin/bash


# install `tbv` library
pip install -e .


# install `mseg-api` anywhere on your machine using:
git clone https://github.com/mseg-dataset/mseg-api.git
cd mseg-api
pip install -e .
cd ..


# install `mseg-semantic` anywhere on your machine using:
git clone https://github.com/mseg-dataset/mseg-semantic.git
pip install -e .
cd ..


# install Eigen
sudo apt install libeigen3-dev


# download `pybind11`
cd tbv-raytracing
git clone https://github.com/pybind/pybind11.git

# without GPU, cannot compile tbv-raytracing lib.