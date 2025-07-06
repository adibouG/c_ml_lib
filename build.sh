#!/bin/bash

set -xe


gcc -O1 -Wall -Wextra  video.c -o video 


gcc -O1 -Wall -Wextra  png2mat.c -o img2mat -I./ext_libs/stb -lm


#clang -Wall -Wextra  adder_generator.c -o adder_generator -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
#./adder_generator
gcc -O1 -Wall -Wextra  ml_viewer.c -o ml_viewer -I./ext_libs/raylib-5.5_linux_amd64/include  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

#./img2mat ./assets/inputs/10031.png imgpixel-3.mat
./img2mat ./assets/inputs/1005.png imgpixel-9.mat
./ml_viewer imgpixel.arch imgpixel-9.mat