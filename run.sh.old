#!/bin/bash

# Regular Colors
black='\e[0;30m'        # Black
red='\e[0;31m'          # Red
green='\e[0;32m'        # Green
yellow='\e[0;33m'       # Yellow
blue='\e[0;34m'         # Blue
purple='\e[0;35m'       # Purple
cyan='\e[0;36m'         # Cyan
white='\e[0;37m'        # White

application=blackscholes

echo -e "${green} CUDA Sobel Edge-Detection Starting... ${white}"

for f in test.data/input/*.data
do
	filename=$(basename "$f")
	filename="${filename%.*}"
	./${application} $f ./test.data/output/${filename}_${application}_nn.data
done
