# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(NOT DEFINED PYTHON_EXECUTABLE)
  execute_process(COMMAND python -c "import sys; print(sys.executable,end='')" OUTPUT_VARIABLE PYTHON_EXECUTABLE)
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" --version OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_PYTHON_VERSION)
  message(STATUS "Using ${TORCH_PYTHON_VERSION}")
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(f'{torch.__version__}',end='')" RESULT_VARIABLE STATUS OUTPUT_VARIABLE TORCH_VERSION)
  if(STATUS AND NOT STATUS EQUAL 0)
    message(FATAL_ERROR "Could not find torch library using python path: ${PYTHON_EXECUTABLE}")
  endif()
  message(STATUS "Found torch ${TORCH_VERSION}")
endif()
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import torch.utils.cpp_extension; print(f'{torch.utils.cpp_extension.CUDA_HOME}',end='')" RESULT_VARIABLE STATUS OUTPUT_VARIABLE CUDA_TOOLKIT_ROOT_DIR)
endif()
if(NOT DEFINED TORCH_LIBRARY_DIRS)
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import torch.utils.cpp_extension; print(';'.join(torch.utils.cpp_extension.library_paths(False)),end='')" COMMAND_ERROR_IS_FATAL ANY OUTPUT_VARIABLE TORCH_LIBRARY_DIRS)
endif()
if(NOT DEFINED TORCH_INCLUDE_DIRS)
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import torch.utils.cpp_extension; print(';'.join(torch.utils.cpp_extension.include_paths(False)),end='')" COMMAND_ERROR_IS_FATAL ANY OUTPUT_VARIABLE TORCH_INCLUDE_DIRS)
endif()

if("$ENV{CUDAARCHS}" STREQUAL "")
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import torch.utils.cpp_extension;print(' '.join(sorted(set(x.split('_')[-1] for x in torch.utils.cpp_extension._get_cuda_arch_flags()))),end='')" COMMAND_ERROR_IS_FATAL ANY OUTPUT_VARIABLE CMAKE_CUDA_ARCHITECTURES)
endif()

# Setup everything
message(STATUS "Using torch libraries: ${TORCH_LIBRARY_DIRS}")
message(STATUS "Using torch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Using CUDA toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_LIBRARY_DIRS})
set(TORCH_LIBRARIES c10 torch torch_cpu torch_python)
string(APPEND CMAKE_CUDA_FLAGS " -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -O3 --use_fast_math")
string(APPEND CMAKE_CXX_FLAGS " -Wno-deprecated-declarations")
