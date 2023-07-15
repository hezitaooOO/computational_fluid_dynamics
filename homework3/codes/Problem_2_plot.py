import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard
from time import sleep
import spatial_operators
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg  # sparse linear algebra
import scipy.linalg as scylinalg               # non-sparse linear algebra
import pylab as plt

RK1 = [0.005349,
0.000452,
7.68E-05,
2.57E-05,
6.76E-05,
7.98E-05]

RK2 = [0.00548,
0.000579,
0.000203,
9.97E-05,
5.75E-05,
4.53E-05]

RK3 = [0.00548,
0.000579,
0.000202,
9.96E-05,
5.75E-05,
4.52E-05]

RK4 = [0.00548,
0.000579,
0.000202,
9.96E-05,
5.75E-05,
4.52E-05]

N_list = [10, 30, 50, 70, 90, 100]

grid_size = [2.*np.pi/x for x in N_list]

order2 = [x**2. for x in grid_size]
order3 = [x**3. for x in grid_size]
order4 = [x**4. for x in grid_size]

plt.plot(grid_size, order2, '--', label = '2nd order')
plt.plot(grid_size, order3, '--', label = '3rd order')

plt.plot(grid_size, order4, '--', label = '4th order')


plt.plot(grid_size, RK1, label = 'RK 1')
plt.plot(grid_size, RK2, label = 'RK 2')
plt.plot(grid_size, RK3, label = 'RK 3')
plt.plot(grid_size, RK4, label = 'RK 4')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Average grid size h')
plt.ylabel('RMS of the error')
plt.legend(loc = 'best')
plt.title('RMS of the error at time t = 1 versus grid size')
plt.savefig('RMS of the error at time t = 1 versus grid size')
plt.xlim()
plt.show()