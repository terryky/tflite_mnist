#!/bin/sh
set -e
set -x

rm -rf checkpoint
rm -rf graph_def
rm -rf saved_model
rm -rf log_data
rm -rf graph_def_export

rm -rf checkpoint_quant
rm -rf graph_def_quant
rm -rf saved_model_quant
rm -rf log_data_quant
rm -rf graph_def_quant_export

rm -f *.tflite

rm -rf LOGDIR
