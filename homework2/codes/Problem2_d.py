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

Nx = 10
Lx = 1.0
CFL = 1.5  # Courant-Friedrichs-Lewy number (i.e. dimensionless time step)
c_x_ref = 10.0
c_x = 1. * c_x_ref  # (linear) convection speed
alpha = 1.0  # diffusion coefficients

m = 1
omega_1 = 2*np.pi/Lx
omega_2 = 2*m*np.pi/Lx
gamma_1 = 1
gamma_2 = 1
c_1 = 13
c_2 = 13

Tf = 1/(omega_2**2*alpha)  # one complete cycle

time_scheme = ["Explicit-Euler", "Explicit-Euler", "Explicit-Euler", "Crank-Nicolson", "Crank-Nicolson", "Crank-Nicolson"]
adv_scheme = ["1st-order-upwind", "2nd-order-central", "2nd-order-upwind", "1st-order-upwind", "2nd-order-central", "2nd-order-upwind"]
dif_scheme = ["2nd-order-central", "2nd-order-central", "2nd-order-central", "2nd-order-central", "2nd-order-central", "2nd-order-central"]  # always-second-order central


# ## Time Advancement
# # time_advancement = "Explicit-Euler"
# time_advancement = "Crank-Nicolson"
#
# ## Advection Scheme
# # advection_scheme = "1st-order-upwind"
# # advection_scheme = "2nd-order-central"
# advection_scheme = "2nd-order-upwind"



def u_initial(X):
    # np.power: First array elements raised to powers from second array, element-wise.
    return c_1*np.sin(omega_1*X-gamma_1)-c_2*np.cos(omega_2*X-gamma_2)  # lambda functions are better..

def u_analytical(x, t):
    return c_1*np.exp(-omega_1**2*alpha*t)*np.sin(omega_1*(x-c_x*t)-gamma_1) - c_2*np.exp(-omega_2**2*alpha*t)*np.cos(omega_2*(x-c_x*t)-gamma_2)


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
dt_max_advective = dx / (c_x + machine_epsilon)  # think of machine_epsilon as zero
dt_max_diffusive = dx2 / (alpha + machine_epsilon)

dt_max = np.min([dt_max_advective, dt_max_diffusive])
dt = CFL * dt_max
# unitary_float = 1.+0.1*machine_epsilon # wat ?!

# Creating identity matrix
Ieye = scysparse.identity(Nx)

# Creating first derivative
# choice of advection_scheme and diffusion_scheme have "2nd-order-central", "2nd-order-upwind",  "1st-order-upwind":
Dx = []
D2x2 = []
for i in xrange(0, 6):
    Dx.append(spatial_discretization.Generate_Spatial_Operators(x_mesh, adv_scheme[i], derivation_order=1))
    # Creating second derivative
    D2x2.append(spatial_discretization.Generate_Spatial_Operators(x_mesh, dif_scheme[i], derivation_order=2))

# Creating A,B matrices such that:
#     A*u^{n+1} = B*u^{n} + q
A = []
B = []
for i in xrange(0, 6):

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
for i in xrange(0, 6):
    A[i], B[i] = scysparse.csr_matrix(A[i]), scysparse.csr_matrix(B[i])

plt.figure(1)
plt.subplot(121)
plt.spy(A[0])
plt.title('EE & 1st upwind A')


plt.subplot(122)
plt.spy(B[0])
plt.title('EE & 1st upwind B')
plt.savefig('EE & 1st upwind AB')
plt.show()



plt.figure(2)
plt.subplot(121)
plt.spy(A[1])
plt.title('EE & 2nd central A')


plt.subplot(122)
plt.spy(B[1])
plt.title('EE & 2nd central B')
plt.savefig('EE & 2nd central AB')
plt.show()





plt.figure(3)
plt.subplot(121)
plt.spy(A[2])
plt.title('EE & 2nd upwind A')


plt.subplot(122)
plt.spy(B[2])
plt.title('EE & 2nd upwind B')
plt.savefig('EE & 2nd upwind AB')
plt.show()




plt.figure(4)
plt.subplot(121)
plt.spy(A[3])
plt.title('CN & 1st upwind A')


plt.subplot(122)
plt.spy(B[3])
plt.title('CN & 1st upwind B')
plt.savefig('CN & 1st upwind AB')
plt.show()





plt.figure(5)
plt.subplot(121)
plt.spy(A[4])
plt.title('CN & 2nd central A')


plt.subplot(122)
plt.spy(B[4])
plt.title('CN & 2nd central B')
plt.savefig('CN & 2nd central AB')
plt.show()


plt.figure(6)
plt.subplot(121)
plt.spy(A[5])
plt.title('CN & 2nd upwind A')


plt.subplot(122)
plt.spy(B[5])
plt.title('CN & 2nd upwind B')
plt.savefig('CN & 2nd upwind AB')
plt.show()