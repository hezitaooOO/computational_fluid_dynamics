# trapSerial.py
# example to run: python trapSerial.py 0.0 1.0 10000


import numpy as np
# import matplotlib.pyplot as plt
# import sys
from mpi4py import MPI
import time

# takes in command-line arguments [a,b,n]
a = -10.0
b = 10.0
order=8
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

dx = (b - a) / (n - 1)
x_mesh = np.linspace(a, b, n)

local_n = int(n / size)
extra_point = n - size * local_n

# we calculate the interval that each process handles
# local_a is the starting point and local_b is the endpoint
local_a = a + rank * local_n * dx
local_b = local_a + (local_n - 1) * dx

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


#####################################



if rank == (size - 1):

   local_integral = integrateRange(local_a, local_b + extra_point * dx, local_n + extra_point)
   print local_integral
   print "when rank == size -1 ", local_integral

else:
   local_integral = integrateRange(local_a, local_b, local_n)
   print "when rank = ", rank, "the local integral is ", local_integral

comm.Barrier()
end_time = time.time()
approximate_integral = comm.reduce(local_integral, op=MPI.SUM, root=0)
time_total = comm.reduce(end_time - start_time, op=MPI.SUM, root=0)

if rank == 1:
   unit_time = np.zeros(1)
   unit_time[0] = end_time-start_time
   np.savetxt('unit_time1.txt', np.transpose(unit_time), fmt='%.18g', delimiter=' ')  ####change here####
   print "unit run time is", unit_time


if rank == 0:
   temp = time_total
   time_total=np.zeros(1)
   time_total[0]=temp
   print "total run time is", time_total

   print "With N =", n, "points, our estimate of the integral from", a, "to", b, "is", approximate_integral
   print "true value of integral is ", true_integral

   np.savetxt('total_time1.txt', np.transpose(time_total), fmt='%.18g', delimiter=' ')	####change here####

   # plt.loglog(inverse_spacing, error, linestyle="-", label="Truncation error")
   # plt.loglog(inverse_spacing, [x ** -1 for x in inverse_spacing], linestyle="-.", label='1st order')
   # plt.loglog(inverse_spacing, [x ** -2 for x in inverse_spacing], linestyle="-.", label='2nd order')
   # plt.loglog(inverse_spacing, [x ** -3 for x in inverse_spacing], linestyle="-.", label='3rd order')
   # plt.loglog(inverse_spacing, [x ** -4 for x in inverse_spacing], linestyle="-.", label='4th order')
   #
   # plt.savefig(("1.png"))
   # plt.show()







