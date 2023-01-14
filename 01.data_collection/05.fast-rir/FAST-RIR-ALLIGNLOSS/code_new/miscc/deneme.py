import os
import errno
import numpy as np


x = np.arange(50)
x=  np.reshape(x, (5,10))

shift=np.array([3,-2,5,-4,7])

for rowNumber in range(x.shape[0]):
    x[rowNumber]=np.roll(x[rowNumber],shift[rowNumber])

print(x)
'''
a=np.arange(2*3*5).reshape(2, 3, 5)

rows = np.array([[0], [1]], dtype=np.intp)
cols = np.array([[2, 3], [1, 2]], dtype=np.intp)

aa = np.stack(a[rows, :, cols]).swapaxes(1, 2)
print(aa)

print(shift)
axis=np.full(shift.shape, 1)
print(axis)
print(x)
print("---")
print(np.roll(x,tuple(shift),tuple(axis)))
'''
