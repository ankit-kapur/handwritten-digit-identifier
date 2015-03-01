# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

train_label = np.array([0,0,1])
x = np.repeat(np.array([2]), 5, 0)
train_label = np.append(train_label, x, 0)
print train_label