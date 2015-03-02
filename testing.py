# -*- coding: utf-8 -*-
import numpy as np
    
matrix = np.array([[0,1,2,5],[0,5,2,8],[0,5,4,9], [0,5,2,1], [0,6,8,3]])

z = np.empty([0])
print "z-initially", z

for i in range(5): 
    zj = np.array([0])         
    for h in range(3):
        zj[0] += 4
    z = np.append(z, zj, 0)

print "z", z
print "z[0]", z[0]
print "z[1]", z[1]
