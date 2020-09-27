# Deep-Learning

A deep-learning repository following IBM's "Deep Learning with Tensorflow" course on edx.

https://courses.edx.org/courses/course-v1:IBM+DL0120EN+2T2020/course/


## Installing TensorFlow

We begin by installing TensorFlow version 2.2.0 and its required prerequistes

```
!pip install grpcio==1.24.3
!pip install tensorflow==2.2.0
```

## Importing TensorFlow

Throughout this repository we will import as 'tf'

```
import tensorflow as tf
if not tf.__version__ == '2.2.0':
    print(tf.__version__)
    raise ValueError('please upgrade to TensorFlow 2.2.0, or restart your Kernel')
```

## Notes

- Note that if you don't have a CUDA-enabled gpu you will get warnings about the cuda library, which You can ignore.