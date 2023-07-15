import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard
from time import sleep
import spatial_discretization
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg  # sparse linear algebra
import scipy.linalg as scylinalg  # non-sparse linear algebra
import pylab as plt
from matplotlib import rc as matplotlibrc

machine_epsilon = np.finfo(float).eps

### this example script is hard-coded for periodic problems

#########################################
############### User Input ##############

Nx = 50
Lx = 1.0
CFL = 0.8  # Courant-Friedrichs-Lewy number (i.e. dimensionless time step)
c_x_ref = 10
c_x = 1. * c_x_ref  # (linear) convection speed
alpha = 0.3  # diffusion coefficients

m = 1
omega_1 = 2*np.pi/Lx
omega_2 = 2*m*np.pi/Lx
gamma_1 = 1
gamma_2 = 1
c_1 = 5
c_2 = 3

Tf = 1/(omega_2**2*alpha)  # one complete cycle
Tf = Tf
time_scheme = ["Crank-Nicolson"] * 3
adv_scheme = ["1st-order-upwind", "2nd-order-central", "2nd-order-upwind"]
dif_scheme = ["2nd-order-central"] * 3

def u_initial(X):
    # np.power: First array elements raised to powers from second array, element-wise.
    return c_1*np.sin(omega_1*X-gamma_1)-c_2*np.cos(omega_2*X-gamma_2)  # lambda functions are better..

def u_analytical(x, t):
    return c_1*np.exp(-omega_1**2*alpha*t)*np.sin(omega_1*(x-c_x*t)-gamma_1) - c_2*np.exp(-omega_2**2*alpha*t)*np.cos(omega_2*(x-c_x*t)-gamma_2)

count = 0

RMS1 = []
RMS2 = []
RMS3 = []
T = list()
dt = 0.1


while dt >= 0.00001:

    count += 1

    #########################################
    ######## Preprocessing Stage ############
    # mesh points with two virtual points at the ends
    xx = np.linspace(0., Lx, Nx + 1)
    # actual mesh points are off the boundaries x=0, x=Lx
    # non-periodic boundary conditions created with ghost points
    x_mesh = 0.5 * (xx[:-1] + xx[1:])

    dx = np.diff(xx)[0]
    dx2 = dx * dx

    # for linear advection/diffusion time step is a function
    # of c,alpha,dx only; we use reference limits, ballpark
    # estimates for Explicit Euler
    # dt_max_advective = dx / (c_x + machine_epsilon)  # think of machine_epsilon as zero
    # dt_max_diffusive = dx2 / (alpha + machine_epsilon)
    #
    # dt_max = np.min([dt_max_advective, dt_max_diffusive])
    # unitary_float = 1.+0.1*machine_epsilon # wat ?!

    # Creating identity matrix
    Ieye = scysparse.identity(Nx)

    # Creating first derivative
    # choice of advection_scheme and diffusion_scheme have "2nd-order-central", "2nd-order-upwind",  "1st-order-upwind":
    Dx = []
    D2x2 = []
    A = []
    B = []
    for i in xrange(0, 3):
        Dx.append(spatial_discretization.Generate_Spatial_Operators(x_mesh, adv_scheme[i], derivation_order=1))
        # Creating second derivative
        D2x2.append(spatial_discretization.Generate_Spatial_Operators(x_mesh, dif_scheme[i], derivation_order=2))

    for i in xrange(3):
        if time_scheme[i] == "Explicit-Euler":
           A.append(Ieye)
           B.append(Ieye - dt * c_x * Dx[i] + dt * alpha * D2x2[i])
        if time_scheme[i] == "Crank-Nicolson":
           adv_diff_Op = -dt * c_x * Dx[i] + dt * alpha * D2x2[i]
           A.append(Ieye - 0.5 * adv_diff_Op)
           B.append(Ieye + 0.5 * adv_diff_Op)

    # plt.spy(Dx)
    # plt.show()

    # forcing csr(Compressed Sparse Row matrix) ordering..
    for i in xrange(3):
        A[i], B[i] = scysparse.csr_matrix(A[i]), scysparse.csr_matrix(B[i])
    u = [u_initial(x_mesh)] * 3  # initializing solution

    time = 0.
    it = 0

    while time < Tf:

       it   += 1
       time += dt
       print "The ", count, "th iteration.", "time/Tf= ", "{0:.0f}%".format(time/Tf * 10), it

       # Update solution
       # solving : A*u^{n+1} = B*u^{n} + q
       # where q is zero for periodic and zero source terms
       for i in xrange(3):
           u[i] = spysparselinalg.spsolve(A[i], B[i].dot(u[i]))   # the first u is initial condition
       # this operation is repeated many times.. you should
       # prefactorize 'A' to speed up computation.
       # ~x is to compute -x-1. so ~bool is either -1 or -2

       if it == int(Tf/dt):
            error1 = np.subtract(u[0], u_analytical(x_mesh, time))
            error2 = np.subtract(u[1], u_analytical(x_mesh, time))
            error3 = np.subtract(u[2], u_analytical(x_mesh, time))

            rms1 = np.sqrt(np.mean(error1**2))
            rms2 = np.sqrt(np.mean(error2**2))
            rms3 = np.sqrt(np.mean(error3**2))


            RMS1.append(rms1)
            RMS2.append(rms2)
            RMS3.append(rms3)

            T.append(1/dt)
    dt = dt/1.5

print T
print RMS1
order1 = [(1/x) ** 1 for x in [float(i) for i in T]]
order2 = [(1/x) ** 2 for x in [float(i) for i in T]]
order3 = [(1/x) ** 3 for x in [float(i) for i in T]]
order4 = [(1/x) ** 4 for x in [float(i) for i in T]]

plt.loglog(T, RMS1, label = "1st-order-upwind")
plt.loglog(T, RMS2, label = "2nd-order-central")
plt.loglog(T, RMS3, label = "2nd-order-upwind")

plt.loglog(T, order1, '--', label = '1st order reference')
plt.loglog(T, order2, '--', label = '2nd order reference')
plt.loglog(T, order3, '--', label = '3th order reference')
# plt.loglog(T, order4, '--', label = '4th order reference')
plt.xlabel('The inverse of timestep (1/dt)')
plt.ylabel('RMS')
plt.title('RMS of different spatial scheme with Crank-Nicolson time scheme')
plt.legend(loc = 'lower right')
plt.savefig('Problem2_c_constant dx.png')
plt.show()

