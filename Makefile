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
GCC			:= g++
CUDA_PATH	:= /usr/local/cuda
NVCC 		:= $(CUDA_PATH)/bin/nvcc -ccbin $(GCC)
NVCC_ARCH	:= -gencode arch=compute_60,code=sm_60 \
				-gencode arch=compute_61,code=sm_61 \
				-gencode arch=compute_62,code=[sm_62,compute_62] \
				-gencode arch=compute_70,code=[sm_70,compute_70]

TARGET		:= blackscholes

################################################################################

all: DIR build

DIR:
	mkdir -p obj

build: $(TARGET)

./obj/BlackScholes_nn.o: ./src/BlackScholes.cu
	$(NVCC) -O3 $(NVCC_ARCH) -o $@ -c $<

./obj/BlackScholes_gold_nn.o:./src/BlackScholes_gold.cpp
	$(NVCC) -O3 $(NVCC_ARCH) -o $@ -c $<

$(TARGET): ./obj/BlackScholes_nn.o ./obj/BlackScholes_gold_nn.o
	$(NVCC) -O3 $(NVCC_ARCH) -o $@ $+

clean:
	rm -rf $(TARGET)
	rm -rf ./obj/BlackScholes_nn.o ./obj/BlackScholes_gold_nn.o

clobber: clean

copy_titanV:
	rsync -av -e ssh --exclude='.git' ./ gpu_carol_titanV211:blackscholes

copy_p100:
	rsync -av -e ssh --exclude='.git' ./ gppd:blackscholes

test:
	./$(TARGET) ./test.data/input/blackscholes_4000K.data ./test.data/output/blackscholes_4000K_blackscholes_nn.data