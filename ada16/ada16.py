#ada16
#https://github.com/zyfang/AdaLovelace2016

import pylab as pl
import numpy as np
import scipy as sp


def ada(watermelon_size):
    remaining = watermelon_size * 0.8
    print remaining
    return remaining


ada_data = np.genfromtxt('ada_data_short.csv', dtype=None, names=True, delimiter=',')

pl.scatter(ada_data["NCD_0"],ada_data["AI_0"])
pl.show()

print ada_data.size




