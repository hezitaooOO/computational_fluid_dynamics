# trapSerial.py
# example to run: python trapSerial.py 0.0 1.0 10000


import numpy as np
from mpi4py import MPI
import time

# takes in command-line arguments [a,b,n]
a = -10.0
b = 10.0

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
order = 9
inverse_spacing=np.zeros(order-1)
error = np.zeros(order-1)

for i in range(2, order+1):

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
    local_b = local_a + local_n * dx

    true_integral = -1 / np.exp(10. ,dtype = np.float128) + np.exp(10., dtype = np.float128) + 2 * np.arctan(1 / np.exp(10., dtype = np.float128), dtype = np.float128) - 2 * np.arctan(np.exp(10., dtype = np.float128), dtype = np.float128)


    #######################################################################################
    def f(x):
        return np.exp(x, dtype = np.float128) * np.tanh(x, dtype = np.float128)


    def integrateRange(a, b, n):
        # n is the number of points(include start point and endpoint)
        dx = ((b - a) / (n - 1))  # dx is the spacing
        grid = np.linspace(a, b, n)
        integral = np.sum(f(grid[:-1]), dtype = np.float128)+np.sum(f(grid[1:]), dtype = np.float128)+4*np.sum(f((grid[:-1]+grid[1:])/2), dtype = np.float128)
        integral = integral * dx / 6

        return integral


    #####################################



    if rank == (size - 1):

        local_integral = integrateRange(local_a, local_b + (extra_point-1) * dx, local_n + extra_point)
        print local_integral
        print "when rank == size -1 ", local_integral

    else:
        local_integral = integrateRange(local_a, local_b, local_n)
        print "when rank = ", rank, "the local integral is ", local_integral

    comm.Barrier()
    end_time = time.time()
    approximate_integral = comm.reduce(local_integral, op=MPI.SUM, root=0)
    time_total = comm.reduce(end_time - start_time, op=MPI.SUM, root=0)

    if rank == 0:
        abs_error = np.abs(np.float128(true_integral) - np.float128(approximate_integral))
        error[i-2]= abs_error
        inverse_spacing[i-2]=1/dx
        # time_total = end_time - start_time
        print "the absolute error is ", abs_error
        print "run time in total is", time_total

        print "With N =", n, "points, our estimate of the integral from", a, "to", b, "is", approximate_integral
        print "true value of integral is ", true_integral
        print "inverse_spacing is ", inverse_spacing
        print "error is ", error

if rank == 0:
    np.savetxt('error1.txt', np.transpose(error), fmt='%.18g', delimiter=' ')
