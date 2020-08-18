# Introduction

This is a repo developed based on [TensorRT Open Source Software](https://github.com/NVIDIA/TensorRT), [tag 20.07](https://github.com/NVIDIA/TensorRT/tree/20.07).

This repo is aim to build a more suitable extended TensorRT Library for RoboRock products. 

# What's new?

1. We rewrote the **batchedNNMSPlugin** for default-box-binding multi-task learning network.

# Installation

## Downloading The TensorRT Components And Build It On Jetson Machine
1. #### Download TensorRT OSS sources.
	```bash
	git clone -b tag_20_07 https://github.com/roborocklsm/TensorRT.git TensorRT
	cd TensorRT
	git submodule update --init --recursive
	```

2. #### Download the TensorRT binary release.
	Since the Jetson Machine is host by Nvdia SDK Manager, the TensorRT binary release is already installed on it. The library directory is usually `/usr/lib/aarch64-linux-gnu/`.

3. #### Build TensorRT OSS sources and copy it into your library.
	```
	export TRT_SOURCE='pwd'
	export TRT_RELEASE=/usr/lib/aarch64-linux-gnu/
	mkdir build 
	cd build
	sudo cmake .. -DTRT_LIB_DIR=$TRT_RELEASE -DTRT_OUT_DIR=`pwd`/out -DTRT_PLATFORM_ID=aarch64 -DCUDA_VERSION=10.2 -DCMAKE_C_COMPILER=/usr/bin/gcc
	sudo make -j$(nproc)
	sudo rm -f /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so
	sudo rm -f /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7
	sudo cp ./out/libnvinfer_plugin.so.7.1.3 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7.1.3
	sudo cp ./out/libnvinfer_plugin_static.a /usr/lib/aarch64-linux-gnu/libnvinfer_plugin_static.a
	sudo ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7.1.3 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so
	sudo ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7.1.3 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7
	```

p.s. You may need to update your cmake since the pre-installed version is 3.10 and this repo requires >=3.13. Strongly recommend 3.17(has been tested).

## Downloading The TensorRT Components And Build It On PC.
You could follow the steps introduced in the [original TensorRT github page](https://github.com/NVIDIA/TensorRT).
