#!/bin/sh
set -e
set -x

#tensorboard --logdir log_data&

# nVidia環境で、GPU 無効化して実行したい場合は下の１行を有効にする
#export CUDA_VISIBLE_DEVICES=

# floatでトレーニングするか、quantでトレーニングするか
FLAG_QUANT=false

while getopts qf OPT
do
  case $OPT in
    "q" ) FLAG_QUANT=true  ;;
    "f" ) FLAG_QUANT=false ;;
  esac
done

if $FLAG_QUANT ; then
  echo "-------------- running quant mode ------------------"
  rm -rf saved_model
  mkdir -p HDF5
  python3 mnist_quant_train.py
else
  echo "-------------- running float mode ------------------"
  rm -rf saved_model
  mkdir -p HDF5
  python3 mnist_float_train.py
fi
