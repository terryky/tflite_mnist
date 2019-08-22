# ------------------------------------------------------
# Tensorflow MNIST チュートリアルから抜粋
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist
# ------------------------------------------------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print(tf.__version__)

# 再現性の確保のために乱数シードを固定（数値は何でもよい）
tf.set_random_seed(12345)


# -------------------------------------------------------
#  入力データ (MNIST)
# -------------------------------------------------------
mnist = input_data.read_data_sets("../MNIST_data/")


# -------------------------------------------------------
#  グラフ構築
# -------------------------------------------------------
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, w) + b

y_ = tf.placeholder(tf.int64, [None])

# 従来のcross-entropy計算式は数値演算的に不安定
# tf.reduce_mean(-tf.reduce_sum(y_ * tf.math.log(tf.nn.softmax(y)),
#                               reduction_indices=[1]))
cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# -------------------------------------------------------
#  学習の実行
# -------------------------------------------------------

BATCH_SIZE = 100            # バッチサイズ
NUM_TRAIN = 1000            # 学習回数

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # トレーニング
    for i in range(NUM_TRAIN):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # test-set で学習済みモデルを検証
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('===================================================')
    print('Accuracy(float): %f' % sess.run(accuracy,
                   feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print('===================================================')


    # ----------------------------------
    # グラフモデル、重みの保存
    # ----------------------------------

    # Checkpoint 形式
    saver = tf.train.Saver()
    saver.save(sess, 'checkpoint/mymodel')

    # GraphDef 形式
    tf.train.write_graph(sess.graph.as_graph_def(), 'graph_def', 'model_graph.pbtxt', as_text=True)
    tf.train.write_graph(sess.graph.as_graph_def(), 'graph_def', 'model_graph.pb'   , as_text=False)


    # SavedModel 形式
    builder = tf.saved_model.builder.SavedModelBuilder('./saved_model')
    signature = tf.saved_model.predict_signature_def(inputs={'input': x}, outputs={'output': y})
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={'predict': signature})
    builder.save(as_text=False)

