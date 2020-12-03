#!/bin/bash
eval ${PRELOAD_FLAG} ${BIN_DIR}/blackscholes -input ${BIN_DIR}/test.data/input/blackscholes_4000K.data -goldOutput ${BIN_DIR}/gold_output.data > stdout.txt 2> stderr.txt