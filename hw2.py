from matplotlib import pyplot as plt
import scipy.io as sp
import numpy as np
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal as mn

#number 1

#The mean vector consists of the means of each variable

#univariate
def average(x):
    average = sum(x) / x.size
    return average

#multivariate
def mean(x):
    return np.apply_along_axis(average, axis=0, arr=x)


#the variance-covariance matrix consists of the variances
#of the variables along the main diagonal and the covariances
#between each pair of variables in the other matrix positions.

def cov(c1, c2, d, m):
    n, v = d.shape
    sum = 0
    for i in range(0, n):
        x = d[i][c1] - m[c1]
        y = d[i][c2] - m[c2]
        sum += x * y
    return sum / (n - 1)


def covariance(x):
    n, v = x.shape
    array = np.zeros(shape=(v,v))
    for i in range(0,v):
        for j in range(0,v):
            array[i][j] = cov(i, j, x, mean(x))
    return array


d1  = np.array([[4.0, 2.0, 0.60],
                [4.2, 2.1, 0.59],
                [3.9, 2.0, 0.58],
                [4.3, 2.1, 0.62],
                [4.1, 2.2, 0.63]])

d2  = np.array([[1, 1, 1],
                [1, 2, 1],
                [1, 3, 2],
                [1, 4, 3],
                [2, 4, 3]])

#Write a generic Matlab function1
#to compute the Mahalanobis distance between two arbitrary
#samples x1 and x2 of a given Gaussian distribution with covariance Σ, mean µ, and dimension d.

#(x - mu)T * sigma_i * (x - mu)
def mahalanobis(x, covariance, mean):
    meanDistance = np.subtract(x, mean)
    tmp = np.matmul(meanDistance, inv(covariance))
    result = np.matmul(tmp, meanDistance.transpose())
    return np.sqrt(result)

x1 = np.array([4.2, 2.1, 0.59])
x2 = np.array([4.0, 2.0, 0.60])

mahalanobis(x2, covariance(d1), mean(d1))

def discriminant(x, covariance, mean, prior, dimension):
    maha = (-0.5) * mahalanobis(x, covariance, mean)
    det = (-0.5) * np.log(np.linalg.det(covariance))
    dem = (dimension / -2) * (np.log(2 * np.pi))
    pri = np.log(prior)
    return (maha - dem - det + pri)

#class1 = np.array([[-5.01, -8.12,]])

data = np.array(sp.loadmat("./data_class3.mat")["Data"][0])

classes = []
classes.append(data[0].transpose())
classes.append(data[1].transpose())
classes.append(data[2].transpose())


apriori = np.array([0.6, 0.2, 0.2])
datapoints = np.array([[1, 3, 2], [4, 6, 1], [7, -1, 0], [-2, 6, 5]])

numClass = len(classes)
for dat in range(0, len(datapoints)):
    print(datapoints[dat])
    for c in range(0, numClass):
        d = discriminant(datapoints[dat], covariance(classes[c]), mean(classes[c]), apriori[c], numClass)
        print(f"class: {c}, descriminant: {d}")

#number 2

def generateNormalDist(num, mu, sigma):
    x = []
    for i in range(0, mu.size):
        s = np.random.standard_normal(num)
        x.append(s)
    
    
    l, phi = np.linalg.eig(sigma)
    
    #convert into diagonal square root
    L = np.diag(np.sqrt(l))
    
    #scale the varience
    y = np.matmul(L, x)
    
    data = np.matmul(phi, y)
    
    for i in range(0, data.shape[1]):
        data[0][i] += mu[0]
        data[1][i] += mu[1]
    
    
    return data

def plot_classes(class1, class2, i):
    
    fig, ax = plt.subplots()
    
    ax.scatter(
               class1[0],
               class1[1],
               c="#c1d657",
               label=f"Class {0}",
               alpha=0.5,
               edgecolors="none")
        
               ax.scatter(
                          class2[0],
                          class2[1],
                          c="#57d6b6",
                          label=f"Class {1}",
                          alpha=0.5,
                          edgecolors="none")
               
               ax.legend()
               ax.grid(True)
               plt.savefig(f"hw2_{i}")


def plot_3d(class1, u1, cov1, i):
    
    X, Y = np.meshgrid(class1[0], class1[1])
    xy = np.column_stack([X.flat, Y.flat])
    
    z = mn.pdf(xy, mean=u1, cov=cov1)
    Z = z.reshape(X.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    plt.savefig(f"hw2_3d_{i}")

def graph_gaussian(number_of_samples, u1, u2, cov1, cov2, i):
    # correlated data
    class1 = generateNormalDist(number_of_samples, u1, cov1)
    class2 = generateNormalDist(number_of_samples, u2, cov2)
    plot_classes(class1, class2, i)
    plot_3d(class1, u1, cov1, i)

i = 2
mu1 = np.array([2, 8])
mu2 = np.array([8, 2])
sigma1 = sigma2 = np.array([[4.1, 0],
                            [0, 2.8]])
num = 1000

sigma1[0, 1] = 0.4
sigma1[1, 0] = 0.4

sigma1 = np.array([[2.1, 1.5], [1.5, 3.8]])

class1 = generateNormalDist(num, mu1, sigma1)
class2 = generateNormalDist(num, mu2, sigma2)

plot_classes(class1, class2, i)
plot_3d(class1, mu1, sigma1, i)
