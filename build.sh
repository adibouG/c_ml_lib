#!/bin/bash

set -xe

gcc -O1 -Wall -Wextra  png2mat.c -o img2mat  -lm

#clang -Wall -Wextra  adder_generator.c -o adder_generator -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
#./adder_generator
gcc -O1 -Wall -Wextra  ml_viewer.c -o ml_viewer -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

./img2mat ./10031.png imgpixel.mat
./ml_viewer imgpixel.arch imgpixel.mat