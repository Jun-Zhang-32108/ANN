# -*- coding: utf-8 -*-
# Autoencoder
# Author: Jun Zhang <junzha@kth.se>
# March 5th, 2019
# 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm



trainingData = np.loadtxt("bindigit_trn.csv", delimiter=",")

tmp = trainingData[0,:].reshape(28,28)
plt.imshow(tmp, cmap=cm.gray)
plt.show()