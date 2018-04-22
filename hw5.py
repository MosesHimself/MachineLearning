from matplotlib import pyplot as plt
import scipy.io as sp
import numpy as np
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal as mn
from scipy.io.matlab import loadmat

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

#(x - mu)T * sigma_i * (x - mu)
def mahalanobis(x, covariance, mean):
    meanDistance = np.subtract(x, mean)
    tmp = np.matmul(meanDistance, inv(covariance))
    result = np.matmul(tmp, meanDistance.transpose())
    return np.sqrt(result)

def discriminant(x, covariance, mean, prior, dimension):
    maha = (-0.5) * mahalanobis(x, covariance, mean)
    det = (-0.5) * np.log(np.linalg.det(covariance))
    dem = (dimension / -2) * (np.log(2 * np.pi))
    pri = np.log(prior)
    return (maha - dem + det + pri)

trainingData, testingData = loadmat("./test_train_data_class3.mat")["Data"][0][0]

trainingData = np.array(trainingData[0])
testingData = np.array(testingData[0])

for i in range(trainingData.shape[0]):
    trainingData[i] = np.transpose(trainingData[i])

for i in range(testingData.shape[0]):
    testingData[i] = np.transpose(testingData[i])
for i in range(trainingData[0].shape[0]):
    plt.scatter(trainingData[0][i][0],trainingData[0][i][1], color = '#000080')

for i in range(trainingData[1].shape[0]):
    plt.scatter(trainingData[1][i][0],trainingData[1][i][1], color='#800000')

for i in range(trainingData[2].shape[0]):
    plt.scatter(trainingData[2][i][0],trainingData[2][i][1], color='#008000')

m1 = mean(trainingData[0])
plt.scatter(m1[0],m1[1], color = '#0000FF', marker=r'$\clubsuit$')

m2 = mean(trainingData[1])
plt.scatter(m2[0],m2[1], color = '#FF0000', marker=r'$\clubsuit$')

m3 = mean(trainingData[2])
plt.scatter(m3[0],m3[1], color = '#00FF00', marker=r'$\clubsuit$')


plt.show()

means = np.array([m1, m2, m3])
covs = np.array([covariance(trainingData[0]), covariance(trainingData[1]), covariance(trainingData[2])])


classes = trainingData.shape[0]
dimension = trainingData[0].shape[1]
size = 0
for i in range(classes):
    size += trainingData[i].shape[0]

flattened = np.zeros(2)
training = np.array([], dtype=int)
for i in range(classes):
    for j in range(trainingData[i].shape[0]):
        training = np.append(training, i)
        flat = np.vstack([flattened, trainingData[i][j]])

flattened = flattened[1:len(flat) + 1]
predicted = np.array([], dtype=int)
discriminants = np.zeros((size, classes))

for i, point in enumerate(flattened):
    for j in range(classes):
        m = discriminant(point, covs[j], means[j], (1/3), dimension)
        discriminants[i, j] = m
        #print(m)

predicted = np.argmax(discriminants, axis=1)

confusion = np.zeros((classes, classes), dtype=int)

for t, p in zip(training, predicted):
    confusion[t, p] += 1

acc = (training == predicted).sum() / len(training)
