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
|学習<br>(TF)|変換<br>(TF)|推論<br>(TFLite)| Float train           | Quant-aware train        |
|:----------:|:----------:|:--------------:| :-------------------: | :-----------------------:|
| 1.12       | 1.12       | 1.12           | 0.9165 <br> 31.56[us] | 0.8858 <br>   33.55[us]  |
| 1.12       | 1.12       | 1.13           | 0.9165 <br> 31.70[us] | 0.8858 <br>   33.87[us]  |
| 1.12       | 1.12       | 1.14           | 0.9165 <br> 28.18[us] | 0.8858 <br> **18.69[us]**|


#### TensowFlow 1.13 の学習結果で推論実行
|学習<br>(TF)|変換<br>(TF)|推論<br>(TFLite)| Float train           | Quant-aware train        |
|:----------:|:----------:|:--------------:| :-------------------: | :-----------------------:|
| 1.13       | 1.13       | 1.12           | 0.9165 <br> 63.09[us] | 0.8852 <br>   66.63[us]  |
| 1.13       | 1.13       | 1.13           | 0.9165 <br> 64.65[us] | 0.8852 <br>   66.47[us]  |
| 1.13       | 1.13       | 1.14           | 0.9165 <br> 45.71[us] | 0.8852 <br> **11.87[us]**|



#### TensowFlow 1.14 の学習結果で推論実行
|学習<br>(TF)|変換<br>(TF)|推論<br>(TFLite)| Float train           | Quant-aware train         |Post-train Quant<br>(w/carib)|Post-train Quant<br>(wo/carib)|
|:----------:|:----------:|:--------------:| :-------------------: | :-----------------------: | :--------------------------:| :---------------------------:|
| 1.14       | 1.14       | 1.12           | 0.9165 <br> 31.15[us] | 0.8856 <br>   34.05[us]   | エラー※1                   |                              |
| 1.14       | 1.14       | 1.13           | 0.9165 <br> 31.52[us] | 0.8856 <br>   32.79[us]   | エラー※1                   |                              |
| 1.14       | 1.14       | 1.14           | 0.9165 <br> 28.87[us] | 0.8856 <br> **11.85[us]** | 0.9076 <br> 376.7[ms]       |                              |


※2) Didn't find op for builtin opcode 'FULLY_CONNECTED' version '4'
※3) Post Training 量子化は、「変換」のステップが存在しない。


#### 備考
Quantize-Awareトレーニングした量子化モデルにおいて、TFLite推論実行時の精度がトレーニング結果よりも劣化している。
この理由は下記か？
- Quant-aware トレーニング時は、(FakeQuant ノードにより量子化処理が加わるものの)、あくまでも float 演算なのに対し
- TFLite で quant モデルを推論実行するときは、INT演算するため

Post Training 量子化したモデルでは、精度劣化は殆ど見られないが、処理時間は float 実行時と変わらない。
重みパラメータは量子化されているものの、実際の演算は float で行われているためか？

