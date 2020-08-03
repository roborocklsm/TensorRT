sudo rm -rf /usr/local/cuda
sudo ln -s /usr/local/cuda-10.0-7.4 /usr/local/cuda

sudo rm -f /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
sudo rm -f /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7

sudo cp /media/shl666/DataDisk/gitWork/TensorRT/build/out/libnvinfer_plugin.so.7.1.3 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7.1.3
sudo cp /media/shl666/DataDisk/gitWork/TensorRT/build/out/libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a

sudo ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7.1.3 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
sudo ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7.1.3 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7
