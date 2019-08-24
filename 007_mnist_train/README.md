# 007_mnist_train

## TensorFlowで学習させた結果の accuracy
|                      | float-train | quant-aware-train| post-train-quant|
|---------             | ----        | ----             | ----|
|TensorFlow 1.12(train)| 0.9922      | 0.9917           |     |
|TensorFlow 1.13(train)| X.9165      | X.9160           |     |
|TensorFlow 1.14(train)| 0.9925      | 0.9920           |     |


## TensorFlow Lite で推論実行させた時の accuracy と処理時間
TensorFlowで学習させた結果をtfliteに変換し、
RaspberryPi 3B の TensorflowLiteで推論実行した時の accuracy と処理時間を計測

#### TensowFlow 1.12 で学習させたパラメータ
|学習  |変換  |推論      | float-train           | quant-aware-train         | post-train-quant      |
|:----:|:----:|:--------:| :-------------------: | :-----------------------: | :-------------------: |
|TF1.12|TF1.12|TFLite1.12| 0.9922 <br> 31.39[ms] | 0.9917 <br>  10.85 [ms]   |                       |
|TF1.12|TF1.12|TFLite1.13| 0.9922 <br> 32.41[ms] | 0.9917 <br>  10.83 [ms]   |                       |
|TF1.12|TF1.12|TFLite1.14| 0.9922 <br> 61.61[ms] | 0.9917 <br> **6.190[ms]** |                       |


#### TensowFlow 1.13 で学習させたパラメータ
|学習  |変換  |推論      | float-train           | quant-aware-train         | post-train-quant      |
|:----:|:----:|:--------:| :-------------------: | :-----------------------: | :-------------------: |
|TF1.13|TF1.13|TFLite1.13| X.9165                | 0.8860                    |                       |



#### TensowFlow 1.14 で学習させたパラメータ
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

