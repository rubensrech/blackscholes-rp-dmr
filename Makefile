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

NVCC_ARCH=-gencode arch=compute_70,code=[sm_70,compute_70]

GCC ?= g++

CUDA_PATH := /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(GCC)


# Common includes and paths for CUDA
BASE_DIR        := /home/carol/rubens/axbench-gpu
FANN_INC        := $(BASE_DIR)/parrot.c/src/FannLib # /usr/local/include
FANN_LIB        := $(BASE_DIR)/parrot.c/src/FannLib # /usr/local/lib
PARROT_LIB      := $(BASE_DIR)/parrot.c/src/ParrotLib
PLANG           := $(BASE_DIR)/parrot.c/src/ParrotObserver/plang.py
PARROT_JSON     := $(BASE_DIR)/parrot.c/src/ParrotObserver/ParrotC.json

LIBRARIES :=

include ./findgllib.mk

LIBRARIES += -lGL -lGLU -lX11 -lXi -lXmu

LFLAGS		:= -lFann -lboost_regex -lParrot
HEADERS     := src
INCLUDE 	:= -I${FANN_INC} -I${HEADERS}
LIB			:= -L${FANN_LIB} -L$(PARROT_LIB)
TARGET		:= blackscholes

################################################################################

all: DIR build

DIR:
	mkdir -p obj

build: $(TARGET)

./obj/BlackScholes_nn.o: ./src/BlackScholes.cu
	$(NVCC) -O3 $(NVCC_ARCH) -o $@ -c $< $(INCLUDE) $(LIB) $(LFLAGS)

./obj/BlackScholes_gold_nn.o:./src/BlackScholes_gold.cpp
	$(NVCC) -O3 $(NVCC_ARCH) -o $@ -c $< $(INCLUDE) $(LIB) $(LFLAGS)

$(TARGET): ./obj/BlackScholes_nn.o ./obj/BlackScholes_gold_nn.o
	$(NVCC) -O3 $(NVCC_ARCH) -o $@ $+ $(LIBRARIES) $(INCLUDE) $(LIB) $(LFLAGS)

run: build
	./$(TARGET)

clean:
	rm -rf $(TARGET)
	rm -rf ./obj/BlackScholes_nn.o ./obj/BlackScholes_gold_nn.o

clobber: clean

copy_titanV:
	rsync -av -e ssh --exclude='.git' ./ gpu_carol_titanV:rubens/blackscholes