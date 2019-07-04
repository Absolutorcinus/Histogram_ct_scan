import numpy as np
from matplotlib import pylab as plt
import scipy
from scipy import signal

A = np.fromfile('volume.raw', dtype='int16', sep="")
#I create another copy of dataset where I will apply the second filter
B = np.fromfile('volume.raw', dtype='int16', sep="")

matrix_3d = A.reshape([421, 512,512])



betha_A = A.flatten()
g = np.bincount(betha_A)
h=g
v = np.nonzero(g)[0]
x=v
w = g[v]
y=w

# plot histogram
fig, ax = plt.subplots()
ax.plot(v,w)
ax.set(title='Histogram of frequency of values in CT scan', xlabel='X', ylabel='Y')

# get input points  from 4 clicks on  the figure
point1, point2 , point3 ,point4 = fig.ginput(4)
p1,p2,p3,p4 = point1[0], point2[0],point3[0], point4[0]
# paint selected area in yellow and blue 
ax.axvspan(point1[0], point2[0], color='yellow', alpha=0.5)
ax.axvspan(point3[0], point4[0], color='blue', alpha=0.3)

# creating mask for our Data: get data between two points
mask1 = (v>p1) & (v<p2)
mask2 = (x>p3) & (x<p4)

# test print application of mask on v "data": 
#((v[mask1])[i]) # the required values of data needed each i represents the sample element that needs to be slected in the main dataset

plt.show()

print("**********(MASKS)***********************")
print("Mask1")
print((mask1))
print("Mask2")
print(len(mask2))
print("Mask3")
mask3 = mask1 & mask2
print(len(mask3))
print("*********************************")


# v[mask1])[0]  we get the minimum value of our data after applying the filter
# so A < (v[mask1])[0] any values less than min we switch it to zero
print((v[mask2])[0],(v[mask2])[-1])

B[B > (v[mask2])[-1]] = 0
B[B <(v[mask2])[0]] = 0

print("------------( B )-------------------")
print(B)
print((v[mask2])[0],(v[mask2])[-1])
print(np.min(B),np.max(B))
print(B)
print("-----------------( END (B) )-------------------")

A[A <(v[mask1])[0]] = 0
A[A > (v[mask1])[-1]] = 0

print("------------( A )-------------------")
print(A)
print((v[mask1])[0],(v[mask1])[-1])
print(np.min(A),np.max(A))
print(A)
print("-----------------( END (A) )-------------------")

C=A+B

matrix_3d = C.reshape([421, 512,512])
domain = np.identity(3)
test_2=matrix_3d[420, :, :]
signal.order_filter(test_2, domain, 0)
test_3=matrix_3d[:,500,:]
plt.imshow(test_2, cmap='gray')
plt.show()

data=np.zeros((421,512))
for i in range(0,512):
    a= matrix_3d[:,:,i]
    for j in range(0,421):
        data[j][i]= np.sum(a[j,:])
data=np.rot90(np.rot90(np.rot90(data)))
plt.imshow(data, cmap='gray')
plt.show()

