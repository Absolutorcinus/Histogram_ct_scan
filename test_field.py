import numpy as np
from matplotlib import pylab as plt

A = np.fromfile('volume.raw', dtype='int16', sep="")

B = A
matrix_3d = A.reshape([421, 512,512])


betha_A = A.flatten()
g = np.bincount(betha_A)
v = np.nonzero(g)[0]
w = g[v]

# plot histogram
fig, ax = plt.subplots()
ax.plot(v,w)
ax.set(title='Histogram of frequency of values in CT scan', xlabel='X', ylabel='Y')

# get input from 2 clicks on figure
point1, point2 , point3 ,point4 = fig.ginput(4)
# paint selected area in red
ax.axvspan(point1[0], point2[0], color='yellow', alpha=0.5)
ax.axvspan(point3[0], point4[0], color='blue', alpha=0.3)
print(point1[0],point2[0],point3[0],point4[0])

mask1 = (v>point1[0]) & (v<point2[0])
mask2 = (v>point3[0]) & (v<point4[0])
print(mask1)
print(mask2)


print((v[mask1])) # the required values of data needed
print((v[mask2])) # the second required vaues from our data set


plt.show()

print("*********************************")

mask3 = mask1 & mask2
print(mask3)


print("*********************************")
    
A[A < (v[mask3])[0] ] = 0
A[A > (v[mask3])[-1]] = 0



print("end")

matrix_3d = A.reshape([421, 512,512])
test_1=matrix_3d[0, :, :]
test_2=matrix_3d[420, :, :]
test_3=matrix_3d[:,500,:]

plt.imshow(test_1, cmap='gray')
plt.show()
plt.imshow(test_2, cmap='gray')
plt.show()

