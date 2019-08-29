# ------------------------------------------------------
# Tensorflow MNIST チュートリアルから抜粋
# https://www.tensorflow.org/tutorials/
# ------------------------------------------------------
import tensorflow as tf

print(tf.__version__)


# -------------------------------------------------------
#  入力データ (MNIST)
#    ~/.keras/datasets/mnist.npz にダウンロードされる。
#    下記次元の numpy.ndarray が返る。
#       x_train: [60000 x (28 x 28)], y_train: [60000]
#       x_test : [10000 x (28 x 28)], y_test : [10000]
# -------------------------------------------------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test  = x_test .reshape(-1, 28 * 28) / 255.0


# -------------------------------------------------------
#  モデル構築
# -------------------------------------------------------
model = tf.keras.Sequential()
model.add (tf.keras.layers.Dense (10, input_shape=(784,), activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# -------------------------------------------------------
#  学習の実行
# -------------------------------------------------------

BATCH_SIZE = 100            # バッチサイズ
NUM_EPOCH  = 5              # 学習回数

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH)

loss, acc = model.evaluate(x_test, y_test)

print('===================================================')
print('Accuracy(float): %f' % acc)
print('===================================================')


# ----------------------------------
# グラフモデル、重みの保存
# ----------------------------------

# HDF5 形式
model.save('./HDF5/my_model.h5')
model.save_weights('./HDF5/my_model_weights.h5')

# SavedModel 形式
tf.keras.experimental.export_saved_model(model, './saved_model')

