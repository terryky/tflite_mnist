# 001_mnist_train

## 1) TensorFlowで学習
トレーニングのさせ方として

- 通常のfloatでのトレーニングと
- 量子化のためのQuantization-awareトレーニング

を行う。


#### 実行方法
```
 > ./run_train.sh     　# float トレーニング
 > ./run_train.sh -q  　# Quantization-aware トレーニング
```


#### トレーニング結果の Accuracy

|                      |Float training|Quantization-aware training|
|:--------------------:|:------------:|:-------------------------:|
|TensorFlow 1.12       | 0.9165       | 0.9161                    |
|TensorFlow 1.13       | 0.9165       | 0.9159                    |
|TensorFlow 1.14       | 0.9165       | 0.9157                    |

TensorFlowのバージョンが違っても、トレーニング結果に殆ど差は出ない。


## 2) tflite形式へ変換
```
 > ./run_export.sh    　           # 推論用グラフ生成
 > ./run_convert_to_tflite.sh    　# tflite 形式へ変換
```

## 3) TensorFlow Lite で推論実行
- RaspberryPi 3B の TensorflowLiteで推論実行した時の accuracy と処理時間を計測。
- TensorFlow Lite のバージョン違いによる推論性能の違いも見てみる。

```
 > cd 100_mnist_infer_tflite-cpp
 > ./mnist_infer ../001_mnist_train/mnist_frozengraph_float.tflite
 > ./mnist_infer ../001_mnist_train/mnist_frozengraph_quant.tflite
```


#### TensowFlow 1.12 の学習結果で推論実行

|学習  |変換  |推論      | float-train           | quant-aware-train         | post-train-quant      |
|:----:|:----:|:--------:| :-------------------: | :-----------------------: | :-------------------: |
|TF1.12|TF1.12|TFLite1.12| 0.9165 <br> 338.9[ms] | 0.8858 <br> 347.1[ms]     |                       |
|TF1.12|TF1.12|TFLite1.13| 0.9165 <br> 335.9[ms] | 0.8858 <br> 368.5[ms]     |                       |
|TF1.12|TF1.12|TFLite1.14| 0.9165 <br> 305.2[ms] | 0.8858 <br> **154.3[ms]** |                       |

|学習  |変換  |推論      | float-train           | quant-aware-train         | post-train-quant      |
|:----:|:----:|:--------:| :-------------------: | :-----------------------: | :-------------------: |
|TF1.12|TF1.14|TFLite1.12| 0.9165 <br> 341.4[ms] | 0.8858 <br> 370.5[ms]     |                       |
|TF1.12|TF1.14|TFLite1.13| 0.9165 <br> 363.4[ms] | 0.8858 <br> 383.0[ms]     |                       |
|TF1.12|TF1.14|TFLite1.14| 0.9165 <br> 311.6[ms] | 0.8858 <br> **156.6[ms]** |                       |


#### TensowFlow 1.13 の学習結果で推論実行
|学習  |変換  |推論      | float-train           | quant-aware-train         | post-train-quant      |
|:----:|:----:|:--------:| :-------------------: | :-----------------------: | :-------------------: |
|TF1.13|TF1.13|TFLite1.13| 0.9165                | 0.8860                    |                       |



#### TensowFlow 1.14 の学習結果で推論実行
|学習  |変換  |推論      | float-train           | quant-aware-train         | post-train-quant      |
|:----:|:----:|:--------:| :-------------------: | :-----------------------: | :-------------------: |
|TF1.14|TF1.12|TFLite1.12| 0.9165 <br> 361.5[ms] | 0.8856 <br>   383.0[ms]   | ※3                   |
|TF1.14|TF1.12|TFLite1.13| 0.9165 <br> 350.4[ms] | 0.8856 <br>   378.9[ms]   | ※3                   |
|TF1.14|TF1.12|TFLite1.14| 0.9165 <br> 313.6[ms] | 0.8856 <br> **150.1[ms]** | ※3                   |

|学習  |変換  |推論      | float-train           | quant-aware-train         | post-train-quant      |
|:----:|:----:|:--------:| :-------------------: | :-----------------------: | :-------------------: |
|TF1.14|TF1.14|TFLite1.12| 0.9165 <br> 323.2[ms] | 0.8858 <br>   357.6[ms]   | ※2 エラー            |
|TF1.14|TF1.14|TFLite1.13| 0.9165 <br> 346.2[ms] | 0.8858 <br>   386.4[ms]   | ※2 エラー            |
|TF1.14|TF1.14|TFLite1.14| 0.9165 <br> 312.7[ms] | 0.8858 <br> **156.9[ms]** | 0.9076 <br> 376.7[ms] |


※2) Didn't find op for builtin opcode 'FULLY_CONNECTED' version '4'
※3) Post Training 量子化は、「変換」のステップが存在しない。


#### 備考
Quantize-Awareトレーニングした量子化モデルにおいて、TFLite推論実行時の精度がトレーニング結果よりも劣化している。
この理由は下記か？
- Quant-aware トレーニング時は、(FakeQuant ノードにより量子化処理が加わるものの)、あくまでも float 演算なのに対し
- TFLite で quant モデルを推論実行するときは、INT演算するため

Post Training 量子化したモデルでは、精度劣化は殆ど見られないが、処理時間は float 実行時と変わらない。
重みパラメータは量子化されているものの、実際の演算は float で行われているためか？

