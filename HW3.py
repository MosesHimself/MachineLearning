from matplotlib import pyplot as plt
import scipy.io as sp
import numpy as np

#part 1
def bayesianBeliefNetwork(a, b, x, c, d):

    #probability of each season
    pa = np.array([.25, .25, .25, .25])

    #p of nAtlantic, p of sAtlantic
    pb = np.array([0.6, 0.4])

    #probability of salmon(row is season, col is locale)
    px1ab = np.array([[0.5, 0.7],
                     [0.6, 0.8],
                     [0.4, 0.1],
                     [0.2, 0.3]])

    #probability of sea bass(row is season, col is locale)
    px2ab = np.array([[0.5, 0.3],
                     [0.4, 0.2],
                     [0.6, 0.9],
                     [0.8, 0.7]])

    #p of c given salmon(light, med, dark)
    pcx1 = np.array([0.6, 0.2, 0.2])
    #p of c given seabass(light, med, dark)
    pcx2 = np.array([0.2, 0.3, 0.5])

    #p of d given salmon(wide, thin)
    pdx1 = np.array([0.3, 0.7])
    #p of d given seabass(wide, thin)
    pdx2 = np.array([0.6, 0.4])

    if x == 0:
        return pa[a] * pb[b] * px1ab[a][b] * pcx1[c] * pdx1[d]

    if x == 1:
        return pa[a] * pb[b] * px2ab[a][b] * pcx2[c] * pdx2[d]

bayesianBeliefNetwork(2, 0, 1, 2, 1)
