import subprocess
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./")

subprocess.call(['gunzip', 't10k-images-idx3-ubyte.gz'])
subprocess.call(['gunzip', 't10k-labels-idx1-ubyte.gz'])
subprocess.call(['gunzip', 'train-images-idx3-ubyte.gz'])
subprocess.call(['gunzip', 'train-labels-idx1-ubyte.gz'])
