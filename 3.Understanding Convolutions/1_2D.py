# Disable debbuging logs (to get rid of cuda warnings)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy import signal as sg

I = [[255, 7, 3],
     [212, 240, 4],
     [218, 216, 230]]

g = [[-1,1]]

print('Without zero padding \n')
print('{0} \n'.format(sg.convolve(I, g, 'valid')))
'''
output:
[[248   4]
 [-28 236]
 [  2 -14]] 
'''

'''
The 'valid' argument states that the output consists
only of those elements that do not rely on the
zero-padding
'''

print ('With zero padding \n')
print ('{0} \n'.format(sg.convolve(I, g, 'full')))

'''
output
[[-255  248    4    3]
 [-212  -28  236    4]
 [-218    2  -14  230]]
'''

print ('Without zero padding_same \n')
print ('{0} \n'.format(sg.convolve( I, g, 'same')))

'''
This output is full discrete linear convultion of the inputs.
It will use zero to complete the input matrix
'''
