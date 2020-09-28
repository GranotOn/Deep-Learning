# Disable debbuging logs (to get rid of cuda warnings)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open('bird.jpg')

# convert("L") translates color images into b/w
image_gr = image.convert("L")

print("\n Original type: %r \n \n" % image_gr) # Original type: <PIL.Image.Image image mode=L size=1920x1440 at 0x247634E89D0>

# convert image to a matrix with values from 0 to 255 (uint8) 
arr = np.asarray(image_gr) 
print("After conversion to numerical representation: \n\n %r" % arr) 

### Plot image

imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')  #you can experiment different colormaps (Greys,winter,autumn)
print("\n Input image converted to gray scale: \n")

# plt.show(imgplot) I have no idea why this isn't showing

'''
Now we can use an edge detector kernel
'''

kernel = np.array([[0,1,0],
                    [1, -4, 1],
                    [0, 1, 0],
                    ])

grad = signal.convolve2d(arr, kernel, mode="same", boundary="symm")

print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')

'''
If we change the kernel and start to analyze the outputs we would
be acting as a CNN. The difference is that a NN do all this work automaticcaly,
as in the kernel adjustment using different weights.
In addition, we can understand how biases affect the behaviour of feature maps.

Please not that when you are dealing with most of the real applications
of CNNs, you usually convert the pixels values to a range from 0 to 1. 
This process is called normalization.
'''

grad_biases = np.absolute(grad) + 100
grad_biases[grad_biases > 255] = 255

print("GRADIENT MAGNITUDE - Feature map")

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad_biases), cmap='gray')

