
# Implementation of Exact Volumetric Ellipsoid Rendering for Real-time View Synthesis

This is not an officially supported Google product.


# Install
Install optix 7.6
Install the usual python library (maybe just use the gaussian splatting env)  
Download slangc https://github.com/shader-slang/slang/releases, tar -xvf it and add it to your bashrc, validate by command "slangc -h"  
You may need to run `export CC=/usr/bin/gcc-11 && export CXX=/usr/bin/g++-11`  
If you encounter issues with nvrtc, nvrtc must be discoverable from CUDA_PATH or PATH.
```
sudo apt-get install libcgal-dev  libgl1-mesa-dev xorg-dev libglm-dev libnvoptix1 
mkdir build
cd build
cmake -DOptiX_INSTALL_DIR=~/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64  ..
make -j16
./bin/splinetracer --file {path to .ply file}
cd ..
pip install slangpy
python train.py
```

# Workaround for Slang NVRTC
Create this file in `/opt/optix/preamble.h`.
```
#define SLANG_CUDA_ENABLE_OPTIX
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
```
Comment out the COMMAND in `configure_slang.cmake` and uncomment one of the alternate commands that directly uses nvcc
```
# COMMAND ${SLANGC} ${slang_file} -o ${intermediate_file1} -dump-intermediates -line-directive-mode none && cat /opt/optix/preamble.h ${intermediate_file1} > ${intermediate_file2} && nvcc -I${OptiX_INSTALL_DIR}/include -lineinfo --ptx ${intermediate_file2} -ccbin /usr/bin/gcc-11
```

# Notes to Devs
There are 2 sets of global mutable variables in `GAS.cpp` and `py_binding.cpp`.
They make the program more efficient to save from allocating the same memory over and over again, but mean that 2 operations shouldn't overlap in execution.
Both Gaussian splatting and this use Shuster quaternion convention, which means the rotation matrices are transposed. Likely reason behind this is the tranposed matrix initialization. 

