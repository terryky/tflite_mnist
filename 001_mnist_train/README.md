# 001_mnist_train

#### accuracy
|                 | float-train | quant-aware-train| post-train-quant|
|---------        | ----        | ----             | ----|
|TensorFlow(train)| 0.916500    | 0.916000         |     |
|TFLite (infer)   | 0.916500    | 0.886000         |     |

Quantization-aware トレーニング時の精度と比較して、TFLite推論実行時の精度が劣化している。この理由は下記？
- Quant-aware トレーニング時は、(FakeQuant ノードにより量子化処理が加わるものの)、あくまでも float 演算なのに対し
- TFLite で quant モデルを推論実行するときは、INT演算するため
