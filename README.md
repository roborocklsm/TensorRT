[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)



# TensorRT Open Source Software

This repository contains the Open Source Software (OSS) components of NVIDIA TensorRT. Included are the sources for TensorRT plugins and parsers (Caffe and ONNX), as well as sample applications demonstrating usage and capabilities of the TensorRT platform.


## Prerequisites

To build the TensorRT OSS components, ensure you meet the following package requirements:

**System Packages**

* [CUDA](https://developer.nvidia.com/cuda-toolkit)
  * Recommended versions:
  * cuda-11.0 + cuDNN-8.0
  * cuda-10.2 + cuDNN-8.0

* [GNU Make](https://ftp.gnu.org/gnu/make/) >= v4.1

* [CMake](https://github.com/Kitware/CMake/releases) >= v3.13

* [Python](<https://www.python.org/downloads/>)
  * Recommended versions:
  * [Python2](https://www.python.org/downloads/release/python-2715/) >= v2.7.15
  * [Python3](https://www.python.org/downloads/release/python-365/) >= v3.6.5

* [PIP](https://pypi.org/project/pip/#history) >= v19.0
  * PyPI packages
  * [numpy](https://pypi.org/project/numpy/)
  * [onnx](https://pypi.org/project/onnx/1.6.0/) 1.6.0
  * [onnxruntime](https://pypi.org/project/onnxruntime/) >= 1.3.0
  * [pytest](https://pypi.org/project/pytest/)

* Essential libraries and utilities
  * [Git](https://git-scm.com/downloads), [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/), [Wget](https://www.gnu.org/software/wget/faq.html#download), [Zlib](https://zlib.net/)

* Cross compilation for Jetson platforms requires JetPack's host component installation
  * [JetPack](https://developer.nvidia.com/embedded/jetpack) >= 4.4

* Cross compilation for QNX requires the qnx developer toolchain
  * [QNX](https://blackberry.qnx.com/en)

**Optional Packages**

* Containerized builds
  * [Docker](https://docs.docker.com/install/) >= 19.03
  * [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) >= 2.0 or `nvidia-container-toolkit`

* Code formatting tools
  * [Clang-format](https://clang.llvm.org/docs/ClangFormat.html)
  * [Git-clang-format](https://github.com/llvm-mirror/clang/blob/master/tools/clang-format/git-clang-format)

* Required PyPI packages for Demos
  * [Tensorflow-gpu](https://pypi.org/project/tensorflow/1.14.0/) == 1.15.0

**TensorRT Release**

* [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download) v7.1


NOTE: Along with the TensorRT OSS components, the following source packages will also be downloaded, and they are not required to be installed on the system.

- [ONNX-TensorRT](https://github.com/onnx/onnx-tensorrt) v7.1
- [CUB](http://nvlabs.github.io/cub/) v1.8.0
- [Protobuf](https://github.com/protocolbuffers/protobuf.git) v3.8.x

## Brief Introduction

This repo is based on TensorRT7, tag:20_07, commit head 650d4b655d1b30b39e476f9317445d680f023fbb.

We rewrite the batchedNMSPlugin operation. The nmsIdx is added to 

## Downloading The TensorRT Components And Build It On Jetson Machine
1. #### Download TensorRT OSS sources.
	```bash
	git clone -b master https://github.com/nvidia/TensorRT TensorRT
	cd TensorRT
	git submodule update --init --recursive
	```

2. #### Download the TensorRT binary release.
	Since the Jetson Machine is host by Nvdia SDK Manager, the TensorRT binary release is already installed on it. The library directory is usually `/usr/lib/aarch64-linux-gnu/`.

3. #### Use `build_jeton.sh` build TensorRT OSS sources and copy it into your library automatically.
	```
	sh build_jetson.sh
	```

p.s. You may need to update your cmake since the pre-installed version is 3.10 and this repo requires >=3.15.

## Downloading The TensorRT Components And Build It On PC.
You could follow the steps introduced in the [original TensorRT github page](https://github.com/NVIDIA/TensorRT).

## Useful Resources

#### TensorRT

* [TensorRT Homepage](https://developer.nvidia.com/tensorrt)
* [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
* [TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html)
* [TensorRT Discussion Forums](https://devtalk.nvidia.com/default/board/304/tensorrt/)


## Known Issues

#### TensorRT 7.1
* [demo/BERT](demo/BERT) has a known accuracy regression for Volta GPUs; F1 score dropped (from 90 in TensorRT 7.0) to 85. A fix is underway.
* See [Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/tensorrt-7.html#rel_7-1-3).
