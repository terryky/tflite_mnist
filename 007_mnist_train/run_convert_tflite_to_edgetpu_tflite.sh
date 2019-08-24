#!/bin/sh
set -e
set -x


BASE_DIR=./
#BASE_DIR=./1.14/


SRC_TFLITE=${BASE_DIR}/mnist_frozengraph_quant.tflite

# EdgeTPU用にコンパイル。ログ表示用オプション (-s) を付与
#   --> mnist_quant_edgetpu.tflite が出力される
edgetpu_compiler ${SRC_TFLITE} -s

mv mnist_frozengraph_quant_edgetpu.* ${BASE_DIR}

