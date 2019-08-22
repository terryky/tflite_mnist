#!/bin/sh
set -e
set -x

#tensorboard --logdir log_data&

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
  rm -rf checkpoint_quant
  rm -rf graph_def_quant
  rm -rf saved_model_quant
  python3 mnist_quant_train.py
else
  echo "-------------- running float mode ------------------"
  rm -rf checkpoint
  rm -rf graph_def
  rm -rf saved_model
  python3 mnist_float_train.py
fi
