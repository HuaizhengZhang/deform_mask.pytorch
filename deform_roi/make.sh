#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling my_lib kernels by nvcc..."
#nvcc -c -o deform_roi_kernel.cu.o deform_roi_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

nvcc -c -o deform_roi_kernel.cu.o deform_roi_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52


cd ../
python3 build.py