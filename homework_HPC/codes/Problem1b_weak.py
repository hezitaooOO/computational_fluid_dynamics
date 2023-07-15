# trapSerial.py
# example to run: python trapSerial.py 0.0 1.0 10000


import numpy as np
# import matplotlib.pyplot as plt
# import sys
from mpi4py import MPI
import time

# takes in command-line arguments [a,b,n]
order=7
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

i=order
n=10**i+1
print "when n is ", n

if rank == 0:
   print "rank is ", rank
   print "size is ", size

comm.Barrier()
start_time = time.time()




true_integral = np.float128(-1 / np.exp(10) + np.exp(10) + 2 * np.arctan(1 / np.exp(10)) - 2 * np.arctan(np.exp(10)))


#######################################################################################
def f(x):
   return np.exp(x) * np.tanh(x)


def integrateRange(a, b, n):
   # n is the number of points(include start point and endpoint)
   dx = ((b - a) / (n - 1))  # dx is the spacing

   integral = np.float128(0)  # initial integral
   grid = np.linspace(a, b, n)
   for i in range(0, n - 1):
   	integral = integral + f(grid[i]) + 4 * f(grid[i] / 2 + grid[i + 1] / 2) + f(grid[i + 1])
   # print "integral is ", integral
   integral = integral * dx / 6

   return np.float128(integral)


######################################################################################



approximate_integral = integrateRange(-10, 10, 10**order+1)

comm.Barrier()
end_time = time.time()

if rank == 0:
   unit_time = np.zeros(1)
   unit_time[0] = end_time-start_time
   np.savetxt('weak_unit_time16.txt', np.transpose(unit_time), fmt='%.18g', delimiter=' ')  ####change here####
   print "weak unit run time is", unit_time










