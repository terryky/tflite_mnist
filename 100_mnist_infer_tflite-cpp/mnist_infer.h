#ifndef _MNIST_INFER_H_
#define _MNIST_INFER_H_

int infer_float (tflite::Interpreter *interpreter);
int infer_uint8 (tflite::Interpreter *interpreter);

#endif /* _MNIST_INFER_H_ */

