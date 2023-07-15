# trapSerial.py
# example to run: python trapSerial.py 0.0 1.0 10000


import numpy as np

#import sys
from mpi4py import MPI
import time

# takes in command-line arguments [a,b,n]
a = -10.0
b = 10.0

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
order = 7
i = order
error = np.zeros(1)


n = 10**i+1
print "when n is ", n

comm.Barrier()
start_time = time.time()

dx = (b - a) / (n - 1)
x_mesh = np.linspace(a, b, n)

local_n = int(n / size)
extra_point = n - size * local_n

# we calculate the interval that each process handles
# local_a is the starting point and local_b is the endpoint
local_a = a + rank * local_n * dx
local_b = local_a + local_n * dx

true_integral = -1 / np.exp(10. ,dtype=np.float128) + np.exp(10., dtype=np.float128) + 2 * np.arctan(1 / np.exp(10., dtype=np.float128), dtype=np.float128) - 2 * np.arctan(np.exp(10., dtype = np.float128), dtype = np.float128)


#######################################################################################
####################### Do not change #################################################
def f(x):
    return np.exp(x, dtype=np.float128) * np.tanh(x, dtype=np.float128)


def integrateRange(a, b, n):
    # n is the number of points(include start point and endpoint)
    dx = ((b - a) / (n - 1))  # dx is the spacing
    grid = np.linspace(a, b, n)
    integral = np.sum(f(grid[:-1]), dtype=np.float128)+np.sum(f(grid[1:]), dtype=np.float128)+4*np.sum(f((grid[:-1]+grid[1:])/2), dtype=np.float128)
    integral = integral * dx / 6

    return integral


########################################################################################



if rank == (size - 1):

    local_integral = integrateRange(local_a, local_b + (extra_point-1) * dx, local_n + extra_point)
    print local_integral

else:
    local_integral = integrateRange(local_a, local_b, local_n)

comm.Barrier()
end_time = time.time()
approximate_integral = comm.reduce(local_integral, op=MPI.SUM, root=0)
time_total = comm.reduce(end_time - start_time, op=MPI.SUM, root=0)

if rank == 0:
    abs_error = np.abs(np.float128(true_integral) - np.float128(approximate_integral))
    error[0] = abs_error
    # time_total = end_time - start_time
    print "run time in total is", time_total
    print "error is ", error
    np.savetxt('error.txt', np.transpose(error), fmt='%.18g', delimiter=' ')





