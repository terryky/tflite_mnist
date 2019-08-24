# ------------------------------------------------------
# Tensorflow MNIST チュートリアルから抜粋
# https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_deep.py
# ------------------------------------------------------
import tensorflow as tf


def deepnn(x):

    # CNN用にReshape
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # [CONV1] グレースケール画像を 32 個の特徴マップに割り当て
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        tf.summary.histogram('W_conv1', W_conv1)
        tf.summary.histogram('b_conv1', b_conv1)

    # [Pooling1] 1/2 ダウンサンプル
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # [CONV2] 32 個の特徴マップを 64 個の特徴マップに割り当て
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        tf.summary.histogram('W_conv2', W_conv2)
        tf.summary.histogram('b_conv2', b_conv2)

    # [Pooling2]
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # [FC1] (28x28)画像 ==> (7x7)x64 特徴マップ ==> 1024 特徴マップ
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # [Dropout]
    h_fc1_drop = h_fc1

    # [FC2] 1024特徴マップ ==> 10クラス
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # [Softmax] 確率値に変える。
    # ※MatMulの後ろにFakeQuantが追加されるので、
    #   MatMulがノード終端だと、その後ろのFakeQuantがトレーニング対象外になってしまう
    #   それを避けるため、ノード終端に Softmaxノード(FakeQuant不要) を追加する
    with tf.name_scope('softmax'):
        y_conv = tf.nn.softmax(y_conv)

    # -------------------------------------------------------
    #  Fake Quantized トレーニンググラフ構築
    # -------------------------------------------------------
    g = tf.get_default_graph()
    tf.contrib.quantize.create_eval_graph(input_graph=g)

    return y_conv


# Fullストライドな 2D convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 2X ダウンサンプル
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 指定された shape の Weight用 Variable
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 指定された shape のバイアス用 Variable
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# --------------------------------------
#   アプリメイン
# --------------------------------------
def main(_):

    # Placeholder
    x  = tf.placeholder(tf.float32, [None, 784])

    # グラフ構築
    y_conv = deepnn(x)

    with tf.Session() as sess:

        # ----------------------------------
        # グラフモデル、重みの保存
        # ----------------------------------

        # GraphDef 形式
        tf.train.write_graph(sess.graph.as_graph_def(), 'graph_def_quant_export', 'model_graph.pbtxt', as_text=True)
        tf.train.write_graph(sess.graph.as_graph_def(), 'graph_def_quant_export', 'model_graph.pb'   , as_text=False)


# --------------------------------------
#   エントリポイント
# --------------------------------------
if __name__ == '__main__':
  tf.app.run(main=main)

