#!/bin/bash

clear
#cd ./results
# rm ./results/*.ppm
rm ./results/*.txt
cd ./build
make clean
make
cd ..
./build/ImageD /media/eric/DE80F79A80F7777D/eric_linux_data/out/ $1 $2
