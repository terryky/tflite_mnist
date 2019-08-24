# 007_mnist_train

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
|TensorFlow 1.12       | 0.9922       | 0.9917                    |
|TensorFlow 1.13       | 0.9907       | 0.9921                    |
|TensorFlow 1.14       | 0.9925       | 0.9920                    |

TensorFlowのバージョンが違っても、トレーニング結果に殆ど差は出ない。


## 2) tflite形式へ変換
```
 > ./run_export.sh    　           # 推論用グラフ生成
 > ./run_convert_to_tflite.sh    　# tflite 形式へ変換
```

## 3) TensorFlow Lite で推論実行
RaspberryPi 3B の TensorflowLiteで推論実行した時の accuracy と処理時間を計測

#### TensowFlow 1.12 の学習結果で推論実行
|学習  |変換  |推論      | float-train           | quant-aware-train         | post-train-quant      |
|:----:|:----:|:--------:| :-------------------: | :-----------------------: | :-------------------: |
|TF1.12|TF1.12|TFLite1.12| 0.9922 <br> 31.39[ms] | 0.9917 <br>  10.85 [ms]   |                       |
|TF1.12|TF1.12|TFLite1.13| 0.9922 <br> 32.41[ms] | 0.9917 <br>  10.83 [ms]   |                       |
|TF1.12|TF1.12|TFLite1.14| 0.9922 <br> 61.61[ms] | 0.9917 <br> **6.190[ms]** |                       |


#### TensowFlow 1.13 の学習結果で推論実行
|学習  |変換  |推論      | float-train           | quant-aware-train         | post-train-quant      |
|:----:|:----:|:--------:| :-------------------: | :-----------------------: | :-------------------: |
|TF1.13|TF1.13|TFLite1.13| X.9165                | 0.8860                    |                       |



#### TensowFlow 1.14 の学習結果で推論実行
|学習  |変換  |推論      | float-train           | quant-aware-train         | post-train-quant      |
|:----:|:----:|:--------:| :-------------------: | :-----------------------: | :-------------------: |
|TF1.14|TF1.14|TFLite1.12| 0.9925 <br> 27.52[ms] | 0.9921 <br>  10.91 [ms]   |                       |
|TF1.14|TF1.14|TFLite1.13| 0.9925 <br> 29.61[ms] | 0.9921 <br>  11.14 [ms]   |                       |
|TF1.14|TF1.14|TFLite1.14| 0.9925 <br> 57.56[ms] | 0.9921 <br> **6.234[ms]** |                       |


※2) Didn't find op for builtin opcode 'FULLY_CONNECTED' version '4'
※3) Post Training 量子化は、「変換」のステップが存在しない。


#### 備考
Quantize-Awareトレーニングした量子化モデルにおいて、TFLite推論実行時の精度がトレーニング結果よりも劣化している。
この理由は下記か？
- Quant-aware トレーニング時は、(FakeQuant ノードにより量子化処理が加わるものの)、あくまでも float 演算なのに対し
- TFLite で quant モデルを推論実行するときは、INT演算するため

Post Training 量子化したモデルでは、精度劣化は殆ど見られないが、処理時間は float 実行時と変わらない。
重みパラメータは量子化されているものの、実際の演算は float で行われているためか？
