include ../Makefile.env

EDGETPU_DIR    = $(HOME)/work/edgetpu-native
TENSORFLOW_DIR = $(EDGETPU_DIR)/tensorflow
TFLITE_LIB_DIR = $(TENSORFLOW_DIR)/tensorflow/lite/tools/make/gen/linux_x86_64/lib/

TARGET  = mnist_infer

SRCS =
SRCS += mnist_infer.cpp
SRCS += mnist_infer_float.cpp
SRCS += mnist_infer_uint8.cpp

INCLUDES += -I$(EDGETPU_DIR)/libedgetpu
LDFLAGS  += -L$(EDGETPU_DIR)/libedgetpu
LDFLAGS  += -ledgetpu
LDFLAGS  += -lunwind


include ../Makefile.include

