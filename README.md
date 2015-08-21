# Caffe

Caffe is a deep learning framework developed with cleanliness, readability, and speed in mind.<br />
Consult the [project website](http://caffe.berkeleyvision.org) for all documentation.<br/>

---
## Caffe in Windows
This is not the official [Caffe](https://github.com/BVLC/caffe) but a repository to compile Caffe in Windows with Visual Studio 2013 + CUDA 6.5 + OpenCV 2.4.9.

## Quick Setup
1. Download the repository.
2. Download 3rdparty dependencies from [here](https://drive.google.com/file/d/0B_G5BUend20PRnFhMUlMelFEZW8/view?usp=sharing), and unzip to the root of caffe. Make sure folder '3rdparty', 'bin', 'caffe', 'examples', 'include', ... are in the same path.
3. Install [CUDA toolkit 6.5](https://developer.nvidia.com/cuda-toolkit).
4. Install [OpenCV 2.4.9](https://initialneil.wordpress.com/2014/09/25/opencv-2-4-9-cuda-6-5-visual-studio-2013/) and set up the system variable `OPENCV_X64_VS2013_2_4_9` to the root of OpenCV.
5. Install [Boost 1.56](http://sourceforge.net/projects/boost/files/boost-binaries/1.56.0/boost_1_56_0-msvc-12.0-64.exe/download) and set up the system variable `BOOST_1_56_0` to the root of Boost.
6. Open `caffe/caffe.sln` in Visual Studio 2013 and BUILD. `caffe.exe` will be generated in `caffe/bin`.

## Test in MNIST
1. run `data/mnist/get_mnist.bat` to get MNIST dataset.
2. run `examples/mnist/create_mnist.bat` to convert data to leveldb format.
3. run `examples/mnist/train_lenet.bat` to start training.

---
## How I did this
My step by step record can be found [here](https://initialneil.wordpress.com/2015/01/11/build-caffe-in-windows-with-visual-studio-2013-cuda-6-5-opencv-2-4-9/).