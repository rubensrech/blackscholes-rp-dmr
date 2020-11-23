################################################################################
#
# Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################

# Compilers options
CPP			:= g++
CPPFLAGS	:=

CUDA_PATH	:= /usr/local/cuda
NVCC 		:= $(CUDA_PATH)/bin/nvcc -ccbin $(CPP)
NVCC_ARCH	:= -gencode arch=compute_60,code=sm_60 \
				-gencode arch=compute_61,code=sm_61 \
				-gencode arch=compute_62,code=[sm_62,compute_62] \
				-gencode arch=compute_70,code=[sm_70,compute_70]
NVCCFLAGS	:= -std=c++11 -O3 -Xptxas -v

INCLUDE		:= -I$(CUDA_PATH)/include
LDFLAGS		:= -L$(CUDA_PATH)/lib64  -lcudart  -lcurand

TARGET		:= blackscholes
OBJ_DIR		:= ./obj
SRC_DIR		:= ./src

CPP_FILES=$(wildcard $(SRC_DIR)/*.cpp)
H_FILES=$(wildcard $(SRC_DIR)/*.h)
CU_FILES=$(wildcard $(SRC_DIR)/*.cu)
CUH_FILES=$(wildcard $(SRC_DIR)/*.cuh)
OBJ_FILES=$(addprefix $(OBJ_DIR)/, $(notdir $(CPP_FILES:.cpp=.o)))
OBJ_FILES+=$(addprefix $(OBJ_DIR)/, $(notdir $(CU_FILES:.cu=.cu.o)))

################################################################################

all: DIR build

DIR:
	mkdir -p $(OBJ_DIR)

build: $(TARGET)

# C++
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(H_FILES)
	$(CPP) $(CPPFLAGS) -c $< -o $@ $(INCLUDE)

# CUDA
$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu $(CUH_FILES)
	$(NVCC) $(NVCC_ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE)

# Executable
$(TARGET): $(OBJ_FILES)
	$(CPP) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(INCLUDE)

clean:
	rm -rf $(TARGET)
	rm -rf $(OBJ_DIR)

clobber: clean

copy_titanV:
	rsync -av -e ssh --exclude='.git' ./ gpu_carol_titanV:rubens/blackscholes

copy_p100:
	rsync -av -e ssh --exclude='.git' ./ gppd:blackscholes