# ------------------------------------------------------
# Post Training Quantize
#  https://www.tensorflow.org/lite/performance/post_training_quantization
# ------------------------------------------------------
import sys
import tensorflow as tf
import tensorflow.lite as lite
from tensorflow.examples.tutorials.mnist import input_data


# TensorFlow 1.14 以降じゃないと
# "AttributeError: module 'tensorflow._api.v1.lite' has no attribute 'Optimize'" エラー
if tf.__version__ < "1.14.0":
    print ("This script doesn't support TensorFlow %s." % tf.__version__)
    sys.exit()

graph_def_file = "graph_def_export/frozen.pb"
input_arrays   = ["Placeholder"]
output_arrays  = ["add"]
tflite_file    = "mnist_frozengraph_posttrain_quant.tflite"

# -------------------------------------------------------
#  入力データ (MNIST)
# -------------------------------------------------------
mnist = input_data.read_data_sets("../MNIST_data/")


# -------------------------------------------------------
#  input, activation のダイナミックレンジ検出用データセット
# -------------------------------------------------------
num_calibration_steps = 1
BATCH_SIZE = 1

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    yield [batch_xs]


# -------------------------------------------------------
#  float学習済みの FrozenPB を 量子化tfliteへ変換 
#    (ダイナミックレンジのキャリブレーション有り）
# -------------------------------------------------------
converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.inference_input_type   = tf.uint8
converter.inference_output_type  = tf.uint8
tflite_model = converter.convert()

open(tflite_file, "wb").write(tflite_model)






# -------------------------------------------------------
# キャリブレーション無しの強制量子化tflite生成
#   (重みパラメータは量子化するが、演算はfloatに戻して実行)
# -------------------------------------------------------
converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
open("mnist_frozengraph_posttrain_size_quant.tflite", "wb").write(tflite_model)

