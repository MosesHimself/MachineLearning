from matplotlib import pyplot as plt
import scipy.io as sp
import numpy as np

classNums = 3
featureNums = 3
sampleNums = 10

class1 = np.array([[0.42, -0.087, 0.58],
                  [-0.2, -3.3, -3.4],
                  [1.3, -0.32, 1.7],
                  [0.39, 0.71, 0.23],
                  [-1.6, -5.3, -0.15],
                  [-0.029, 0.89, -4.7],
                  [-0.23, 1.9, 2.2],
                  [0.27, -0.3, -0.87],
                  [-1.9, 0.76, -2.1],
                  [0.87, -1.0, -2.6]])

class2 = np.array([[-0.4, 0.58, 0.089],
                  [-0.31, 0.27, -0.04],
                  [0.38, 0.055, -0.035],
                  [-0.15, 0.53, 0.011],
                  [-0.35, 0.47, 0.034],
                  [0.17, 0.69, 0.1],
                  [-0.011, 0.55, -0.18],
                  [-0.27, 0.61, 0.12],
                  [-0.065, 0.49, 0.0012],
                  [-0.12, 0.054, -0.063]])

class3 = np.array([[0.83, 1.6, -0.014],
                  [1.1, 1.6, 0.48],
                  [-0.44, -0.41, 0.32],
                  [0.047, -0.45,  1.4],
                  [0.28, 0.35, 3.1],
                  [-0.39, -0.48, 0.11],
                  [0.34, -0.079, 0.14],
                  [-0.3, -0.22, 2.2],
                  [1.1, 1.2, -0.46],
                  [0.18, -0.11, -0.49]])

feature1 = np.empty((30,1))
feature2 = np.empty((30,1))
feature3 = np.empty((30,1))
for i in range(sampleNums):
    feature1[i] = class1[i][0]
    feature1[i + 10] = class2[i][0]
    feature1[i + 20] = class3[i][0]
    feature2[i] = class1[i][1]
    feature2[i + 10] = class2[i][1]
    feature2[i + 20] = class3[i][1]
    feature3[i] = class1[i][2]
    feature3[i + 10] = class2[i][2]
    feature3[i + 20] = class3[i][2]


def maximum_likelihood1(x):
    #x = np.transpose(x)
    n = x.size

    mu = sum(x) / n


    total = 0

    for i in range(n):
        val = x[i] - mu
        val *= val
        total += val

    sigma = total / n

    print("mu is:")
    print(mu)
    print("sigma is:")
    print(sigma)


def average(x):
    average = sum(x) / x.size
    return average

#multivariate
def mean(x):
    return np.apply_along_axis(average, axis=0, arr=x)


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


def maximum_likelihood2(x):
    #x = np.transpose(x)
    mu = mean(x)
    sigma = covariance(x)
    print("mu is:")
    print(mu)
    print("sigma is:")
    print(sigma)

#part a
print("----")
print("--feature 1--")
maximum_likelihood1(feature1[:10])

print("----")
print("--feature 2--")
maximum_likelihood1(feature2[:10])

print("----")
print("--feature 3--")
maximum_likelihood1(feature3[:10])

#part b
x1 = np.array([class1[0], class1[1]])
x2 = np.array([class1[0], class1[2]])
x3 = np.array([class1[1], class1[2]])

print("----")
print("--feature 1 and 2--")
maximum_likelihood2(x1)

print("----")
print("--feature 1 and 3--")
maximum_likelihood2(x2)

print("----")
print("--feature 2 and 3--")
maximum_likelihood2(x3)

#part c
maximum_likelihood2(class1)

#part d
maximum_likelihood2(class2)
