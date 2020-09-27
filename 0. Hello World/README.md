# Hello World

We will explore basic TensorFlow syntax and will get familiar with the structure of data.

## Files structure (in order)

- hello_world.py
- arrays.py
- practice.py
- variables.py
- operations.py

## What is the meaning of Tensor?

In TensorFlow all data is passed between operations in a computation graph, and these are passed in the form of Tensors, hence the name of TensorFlow.

The word tensor from new latin means "that which stretches". It is a mathematical object that is named "tensor" because an early application of tensors was the study of materials stretching under tension. The contemporary meaning of tensors can be taken as multidimensional arrays.

![dimensions](https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Dimension_levels.svg/354px-Dimension_levels.svg.png)

- The zero dimension can be seen as a point, or a single object.
- The first dimension can be seen as a line, and expressed using a single-dimensional array. (see a bellow)
- The second dimension can be seen as a surface, and imagined using a matrix. (see b bellow)
- The third dimension can be seen as a volume, a three-dimensional array can be seen as an infinite series of surfaces along an infinite line. (see c bellow)

And so on..

![dimensions_representation](https://c.mql4.com/book/i/59.png)

## Why Tensors?

The tensor structure helps us by giving the freedom to shape the dataset in the way we want.

It is particulary helpful when dealing with images, due to how they are encoded.

Images have height and width, so it is sensible to represent them with a two dimensional structure, such as a matrix. When adding the fact that images have colors (RGB), we need a dimension. That's when Tensors become particulary helpful.

![image_RGB](https://docs.microsoft.com/en-us/windows/win32/wic/graphics/ycbcr1.png)


### Credits

Notebook created by: Saeed Aghabozorgi and Rafael Belo Da Silva

Updated to TF 2.X by Samaya Madhavan

### References 

https://www.tensorflow.org/versions/r0.9/get_started/index.html
http://jrmeyer.github.io/tutorial/2016/02/01/TensorFlow-Tutorial.html
https://www.tensorflow.org/versions/r0.9/api_docs/python/index.html
https://www.tensorflow.org/versions/r0.9/resources/dims_types.html
https://en.wikipedia.org/wiki/Dimension
https://book.mql4.com/variables/arrays
https://msdn.microsoft.com/en-us/library/windows/desktop/dn424131(v=vs.85).aspx