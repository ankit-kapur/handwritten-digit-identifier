# -*- coding: utf-8 -*-
import numpy as np
    
matrix = np.array([[0,1,2,5],[0,5,2,8],[0,5,4,9], [0,5,2,1], [0,6,8,3]])


def generateLabelVector(x):
    vector = np.repeat(np.array([0]), 10, 0)
    vector[x] = 1
    return vector

label_vector = generateLabelVector(3)
print label_vector

repeated_labels_testdata = np.tile(label_vector, (4,1))

print repeated_labels_testdata
