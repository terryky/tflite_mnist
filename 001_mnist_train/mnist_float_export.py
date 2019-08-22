# ------------------------------------------------------
# Tensorflow MNIST チュートリアルから抜粋
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist
# ------------------------------------------------------
import tensorflow as tf

print(tf.__version__)


# -------------------------------------------------------
#  グラフ構築
# -------------------------------------------------------
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, w) + b



with tf.Session() as sess:

    # ----------------------------------
    # グラフモデル、重みの保存
    # ----------------------------------

    # GraphDef 形式
    tf.train.write_graph(sess.graph.as_graph_def(), 'graph_def_export', 'model_graph.pbtxt', as_text=True)
    tf.train.write_graph(sess.graph.as_graph_def(), 'graph_def_export', 'model_graph.pb'   , as_text=False)
