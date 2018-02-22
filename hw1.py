
from matplotlib import pyplot as plt
import scipy.io as sp
import numpy as np


mat_contents = sp.loadmat("data_class4.mat")
data = mat_contents['Data']

#a)
class0 = data[0][0]
class1 = data[0][1]
class2 = data[0][2]
class3 = data[0][3]

mean0 = np.mean(data0)
mean1 = np.mean(data1)
mean2 = np.mean(data2)
mean3 = np.mean(data3)

cov0 = np.cov(data0)
cov1 = np.cov(data1)
cov2 = np.cov(data2)
cov3 = np.cov(data3)

#b)
def eigenvector(a):
    #I have no idea how to do this, I can't teach myself what
    #eigenvectors are and then write this function before this
    #is due tomorrow :(
  return a

#c)
plt.plot(class0[0], class0[1], 'ro')
plt.plot(class1[0], class1[1], 'bs')
plt.plot(class2[0], class2[1], 'g^')
plt.plot(class3[0], class3[1], 'o--')
plt.show()
