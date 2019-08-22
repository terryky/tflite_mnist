#!/bin/bash
set -e


#
# ■TensorFlow の Frozen PB を、floatのまま TFLite に変換する
#
#
# [usage] 
#   > ./convert_pb_to_tflite_float frozen_pb in_node out_node
#
# ./convert_pb_to_tflite_float.sh mobilenet_v1_1.0_224_frozen.pb input MobilenetV1/Predictions/Reshape_1


#
# TensorFlow マニュアル：tensorflow/tensorflow/lite/g3doc/convert/cmdline_reference.md
#
#  --graph_def_file           : 入力グラフ(frozen PB)
#  --saved_model_dir          : 入力グラフ(Saved Model directory)
#  --keras_model_file         : 入力グラフ(Keras Model file)
#
#  --output_format            : TFLITE [デフォルト]
#                               GRAPHVIZ_DOT指定すると可視化情報を出力する
#
#  --mean_values              : 入力テンソルが量子化されているときに必要なオプション
#  --std_dev_values           : RealVal = (QuantInput - mean) / std_dev
#
#                               [inference_type=FLOAT] 
#                                 INT入力は上記変数でfloatに戻されてから推論実行
#                               [inference_type=UINT8] 
#                                 INT入力のまま推論実行。上記変数は 固定少数の乗算値として使用
#
#  --inference_type           : FLOAT [デフォルト] 
#                               QUANTIZED_INT8) 出力ファイル内の実数はUINT8に量子化される
#  --inference_input_type     : デフォルトでは、inference_type と同じ型が使用される
#                               INT8入力だけどFLOATで推論したい場合などに使う
#
#  --default_ranges_min/max   : デフォルトのmin/max値を与えて、強制的に量子化。精度低。実験用
#
#  --drop_control_dependency  : True  [デフォルト]
#                               TFLiteは control dependencies をサポートしないので。
#  --reorder_across_fake_quant: False [デフォルト]
#                               意図しない位置にある FakeQuant ノード並び変えるかどうか
#                               グラフ変換を妨げる位置に FakeQuant がある場合に使う。
#                               グラフ変化するので、演算結果変わる可能性あり
#
#  --allow_custom_ops         : False [デフォルト]
#                               カスタム演算を許可。falseだと未知のOpeはエラーとなる。
#                               カスタム演算には、custom resolver ランタイムを用意する必要がある。
#
#  --post_training_quantizer  : False[デフォルト]
#                               ウェイトを量子化するかどうか。
#
#  --dump_graphviz_dir        : GraphViz の .dot ファイル出力先を指定
#                               変換前/変換後の両方の .dot ファイルを出力
#  --dump_graphviz_video      : 全てのグラフ変換で出力。
#

# 引数が3個でなければエラー
if [ $# -ne 4 ]
then
  echo "[usage] convert_savedmodel_to_tflite_float savedmodel_dir in_node out_node innode_shape"
  exit 0
fi


SAVEDMODEL_DIR=$1
INNODE_NAME=$2
OUTNODE_NAME=$3
INNODE_SHAPE=$4


echo '---------------------------------'
 echo "SavedModel dir   :" $SAVEDMODEL_DIR
 echo "input  node name :" $INNODE_NAME
 echo "output node name :" $OUTNODE_NAME
 echo "input  node shape:" $INNODE_SHAPE
echo '---------------------------------'
set -x


TFLITE_FILE=saved_model.tf_float.tflite

tflite_convert \
  --saved_model_dir=${SAVEDMODEL_DIR} \
  --output_file=${TFLITE_FILE} \
  --output_format=TFLITE \
  --input_arrays=${INNODE_NAME} \
  --output_arrays=${OUTNODE_NAME} \
  --input_shapes=${INNODE_SHAPE} \
  --saved_model_signature_key=predict 


set +x
echo ""
echo "----------------------------------"
echo "[SUCCESS] " ${TFLITE_FILE}
set -x



