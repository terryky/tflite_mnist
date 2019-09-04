# ------------------------------------------------------
# Tensorflow MNIST チュートリアルから抜粋
# https://colab.research.google.com/github/tensorflow/docs/blob/r2.0rc/site/en/r2/tutorials/quickstart/advanced.ipynb
# ------------------------------------------------------
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


# -------------------------------------------------------
#  入力データ (MNIST)
#    ~/.keras/datasets/mnist.npz にダウンロードされる。
#    下記次元の numpy.ndarray が返る。
#       x_train: [60000 x (28 x 28)], y_train: [60000]
#       x_test : [10000 x (28 x 28)], y_test : [10000]
# -------------------------------------------------------
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# 次元を増やす 
# [NWH] (60000, 28, 28) ==> [NWHC] (60000, 28, 28, 1)
x_train = x_train[..., tf.newaxis]
x_test  = x_test [..., tf.newaxis]

# デフォルトの fp64 だと tflite_converter がエラーになるので fp32 にする
x_train = x_train.astype(np.float32)
x_test  = x_test .astype(np.float32)

# numpy array を BatchDataset 型に変換
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test,  y_test )).batch(32)


# -------------------------------------------------------
#  モデル構築 [シンボリック(宣言型)スタイル]
# -------------------------------------------------------
model_symbolic = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense( 10, activation='softmax')
])


# -------------------------------------------------------
#  モデル構築 [モデルサブクラス(命令型)スタイル]
# -------------------------------------------------------
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)

        return x

model_subclass = MyModel()


# どちらのモデルを使うか？
#model = model_symbolic
model = model_subclass



# -------------------------------------------------------
#  Loss, Optimizer, Accuracy 定義
# -------------------------------------------------------
loss_object    = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer      = tf.keras.optimizers.Adam()

train_loss     = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss      = tf.keras.metrics.Mean(name='test_loss')
test_accuracy  = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



# tf.function デコレータ
#   関数のデコレータとして @tf.function 追加すると
#   その関数は実行時にグラフへとコンパイルされる
#
#   関数ごとに動作モードを選択可能
#     Eagerモード
#     Graphモード
#

#@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss        = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


#@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss      = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)



# -------------------------------------------------------
#  学習の実行
# -------------------------------------------------------
EPOCHS = 1

for epoch in range(EPOCHS):
    i = 0
    for images, labels in train_ds:
        train_step(images, labels)
        if (i % 100 == 0):
            print ('training [%d]' % i)
        i += 1

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print('===================================================')
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss    .result(),
                          train_accuracy.result()*100,
                          test_loss     .result(),
                          test_accuracy .result()*100))
    print('===================================================')

    # 次Epoch用に metrics リセット
    train_loss    .reset_states()
    train_accuracy.reset_states()
    test_loss     .reset_states()
    test_accuracy .reset_states()


# ----------------------------------
# グラフモデル、重みの保存
# ----------------------------------

dummy_x = np.zeros((1, 28, 28, 1))
model._set_inputs(dummy_x)


# HDF5 形式

if (model == model_symbolic):
    model.summary()
    model.save('./HDF5/my_model.h5')
    model.save_weights('./HDF5/my_model_weights.h5')


# SavedModel 形式

model.save('./saved_model', save_format='tf')

