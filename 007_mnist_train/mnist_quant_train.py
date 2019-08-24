# ------------------------------------------------------
# Tensorflow MNIST チュートリアルから抜粋
# https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_deep.py
# ------------------------------------------------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



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
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

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
    tf.contrib.quantize.create_training_graph(input_graph=g,
                                              quant_delay=1000)

    return y_conv, keep_prob


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
# 学習済み checkpoint をリストア
# --------------------------------------
def restore_checkpoint(sess, ckpt_dir):

    # 学習済みの checkpoint があるかチェック。なければ何もしない
    ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt_state:
        print(ckpt_state)
    else:
        print("can't find checkpoint directory.")
        return

    ckpt_path = ckpt_state.model_checkpoint_path


    # checkpoint に含まれる Variables のリストを取得
    print ("--------------------------------------------------------")
    print (" Variable list in pretrained \"{}\"".format(ckpt_path))
    print ("--------------------------------------------------------")
    cvar_list = tf.contrib.framework.list_variables(ckpt_dir)
    for cvar in cvar_list:
        print ("{:32}{}".format(cvar[0], cvar[1]))

    # 全Variablesのうち、checkpoint保存済みのものをピックアップ
    print ("--------------------------------------------------------")
    print (" Restore checkpoint parameter to Variables")
    print ("--------------------------------------------------------")
    variables_to_restore = []
    gvar_list = tf.global_variables()
    for gvar in gvar_list:
        restore_flag = False;
        for cvar in cvar_list:
            if gvar.name.split(':')[0] == cvar[0]:
                restore_flag = True
                variables_to_restore.append(gvar)
                break;
        if restore_flag:
            print("[Restore] {}".format(gvar))
        else:
            print("[-------] {}".format(gvar))

    # Saver による復帰
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, ckpt_path)

    print ("--------------------------------------------------------")


# --------------------------------------
#   アプリメイン
# --------------------------------------
def main(_):

    # 再現性の確保のために乱数シードを固定（数値は何でもよい）
    tf.set_random_seed(12345)

    # 入力データ (MNIST)
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

    # Placeholder
    x  = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # グラフ構築
    y_conv, keep_prob = deepnn(x)

    # loss, optimizer
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

    # Tensorboard用ログ出力
    #   host> tensorboard --logdir log_data&
    #   browser> http://libra:6006  
    summary_writer = tf.summary.FileWriter('./log_data_quant')
    summary_writer.add_graph(tf.get_default_graph())
    merged = tf.summary.merge_all()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 学習済みcheckpointを復帰
        restore_checkpoint(sess, 'checkpoint')

        # ----------------------------------
        # トレーニング
        # ----------------------------------
        for i in range(1000):
            batch = mnist.train.next_batch(50)
            feed_train = {x: batch[0], y_: batch[1], keep_prob: 0.5}
            feed_accur = {x: batch[0], y_: batch[1], keep_prob: 1.0}

            # 定期的に Accuracy チェック(Train-set)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict=feed_accur)
                print('step %d, training accuracy %g' % (i, train_accuracy))

                summary_str = sess.run(merged, feed_dict=feed_accur)
                summary_writer.add_summary(summary_str, i)

            # 毎ループトレーニング
            train_step.run(feed_dict=feed_train)


        # トレーニング完了後の Accuracyチェック(test-set)
        acc_val = accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print('===================================================')
        print('Accuracy(quant): %f' % acc_val)
        print('===================================================')


        # ----------------------------------
        # グラフモデル、重みの保存
        # ----------------------------------

        # Checkpoint 形式
        saver = tf.train.Saver()
        saver.save(sess, 'checkpoint_quant/mymodel')

        # GraphDef 形式
        tf.train.write_graph(sess.graph.as_graph_def(), 'graph_def_quant', 'model_graph.pbtxt', as_text=True)
        tf.train.write_graph(sess.graph.as_graph_def(), 'graph_def_quant', 'model_graph.pb'   , as_text=False)


        # SavedModel 形式
        builder = tf.saved_model.builder.SavedModelBuilder('./saved_model_quant')
        signature = tf.saved_model.predict_signature_def(inputs={'input': x}, outputs={'output': y_conv})
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={'predict': signature})
        builder.save(as_text=False)


# --------------------------------------
#   エントリポイント
# --------------------------------------
if __name__ == '__main__':
  tf.app.run(main=main)

