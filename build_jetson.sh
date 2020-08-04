export TRT_SOURCE='pwd'
export TRT_RELEASE=/usr/lib/aarch64-linux-gnu/

mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd`/out
make -j$(nproc)

sudo rm -f /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so
sudo rm -f /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7

sudo cp ./out/libnvinfer_plugin.so.7.1.3 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7.1.3
sudo cp ./out/libnvinfer_plugin_static.a /usr/lib/aarch64-linux-gnu/libnvinfer_plugin_static.a

sudo ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7.1.3 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so
sudo ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7.1.3 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7
