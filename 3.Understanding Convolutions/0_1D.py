# Disable debbuging logs (to get rid of cuda warnings)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

'''
Using the convolution operation between two arrays.
Check: the equation of convultion
'''
# h = [2, 1, 0]
# x = [3, 4, 5]

# y = np.convolve(x, h)
# print(y); # [6, 11, 14, 5, 0]

'''
Now we will experimante with methods of applying a kernel on the matrix.
'''

# 1) Padding(full) method

'''
Think of the kernel as a sliding window.
We have to come with the solution of padding zeros
on the input array. This is a very famous implementation and
will be easier to show how it works with a simple example
'''

# import numpy as np

# x = [6, 2]
# h = [1, 2, 5, 4]

# y = np.convolve(x, h, "full")  #now, because of the zero padding, the final dimension of the array is bigger
# print(y) # 6 14 34 34 8

# 2) Padding (same)

'''
In this approach, we just add the zero to the left (and top of the matrix in 2D).
That is, only the first 4 steps of "full" method.
'''

import numpy as np

# x = [6, 2]
# h = [1, 2, 5, 4]

# y = np.convolve(x, h, "same")  # it is same as zero padding, but with returns an ouput with the same length as max of x or h
# print(y) # 6 14 34 34

# 3) No padding (valid)

'''
In the last case we only applied the kernel when we had
a comptaible position on the h array, in some cases you
want a dimensionality reduction. For this purpose, we ignore the
steps that would need padding (zeros before and after the array)
'''

x = [6, 2]
h = [1, 2, 5, 4]

y = np.convolve(x, h, "valid")
print(y) # 14 34 34

'''
Valid returns output of length max(x, h) - min(x, h) + 1
This is to ensure that values outside of the boundary of 'h'
will not be used in the calculation of the convultion.
In the next example we will understand why we used the argument valid. 
'''

