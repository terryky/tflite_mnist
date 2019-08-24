#!/bin/sh
set -e
set -x

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
  tensorboard --logdir log_data_quant
else
  echo "-------------- running float mode ------------------"
  tensorboard --logdir log_data
fi
