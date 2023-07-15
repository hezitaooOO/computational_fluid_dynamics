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

Nx = 100
Lx = 1.0
CFL = 0.3  # Courant-Friedrichs-Lewy number (i.e. dimensionless time step)
c_x_ref = 10.0
c_x = 1. * c_x_ref  # (linear) convection speed
alpha = 1.0  # diffusion coefficients

m = 1
omega_1 = 2*np.pi/Lx
omega_2 = 2*m*np.pi/Lx
gamma_1 = 1.
gamma_2 = 1.
c_1 = 13.
c_2 = 13.

Tf = 1./(omega_2**2.*alpha)  # one complete cycle

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

#########################################
####### Eigenvalue analysis #############
# T = (scylinalg.inv(A.todense())).dot(B.todense())  # T = A^{-1}*B
# lambdas,_ = scylinalg.eig(T); plt.plot(np.abs(lambdas)); plt.show()
# keyboard()

#########################################
########## Time integration #############

u=[]
for i in xrange(0, 6):
    u.append(u_initial(x_mesh))  # initializing solution

# Figure settings
# matplotlibrc('text.latex', preamble='\usepackage{color}')
# matplotlibrc('text',usetex=True)
# matplotlibrc('font', family='serif')

figwidth = 10
figheight = 6
lineWidth = 4
textFontSize = 28
gcafontSize = 30

plt.ion()  # means interactive mode on
plt.close()

time = 0.
it = 0

ymin = -20.
ymax = 20.
plot_every = 100


print A[0]
print B[0]
print u[0]

print "Tf is", Tf

limit = int(Tf/dt)

while time < Tf:

   it   += 1
   time += dt

   # Update solution
   # solving : A*u^{n+1} = B*u^{n} + q
   # where q is zero for periodic and zero source terms
   for i in xrange(6):
       u[i] = spysparselinalg.spsolve(A[i], B[i].dot(u[i]))   # the first u is initial condition
   # this operation is repeated many times.. you should
   # prefactorize 'A' to speed up computation.
   # ~x is to compute -x-1. so ~bool is either -1 or -2

fig = plt.figure(0, figsize=(figwidth, figheight))
ax   = fig.add_axes([0.15, 0.15, 0.8, 0.8])
plt.axes(ax)
ax.plot(x_mesh, u[0], '-', label = 'EE & 1st-upwind')
ax.plot(x_mesh, u[1], '-', label = 'EE & 2nd-central')
ax.plot(x_mesh, u[2], '-', label = 'EE & 2nd-upwind')
ax.plot(x_mesh, u[3], '-', label = 'CN & 1st-upwind')
ax.plot(x_mesh, u[4], '-', label = 'CN & 2nd-central')
ax.plot(x_mesh, u[5], '-', label = 'CN & 2nd-upwind')


ax.plot(x_mesh, u_initial(x_mesh), '--', label='Initial condition')
ax.plot(x_mesh, u_analytical(x_mesh, time), linestyle="-.", marker = 'o', label = 'Analytical solution')
print "current time is ", time
ax.text(0.7, 0.9, r"$t="+"%1.5f" %time+"$", fontsize=gcafontSize, transform=ax.transAxes)
ax.grid('on', which='both')
plt.setp(ax.get_xticklabels(), fontsize=gcafontSize)
plt.setp(ax.get_yticklabels(), fontsize=gcafontSize)
ax.set_xlabel(r"$x$", fontsize=textFontSize)
ax.set_ylabel(r"$u(x,t)$", fontsize=textFontSize, rotation=90)
plt.legend(loc = 'best')
ax.set_ylim([ymin, ymax])
plt.title('Solution of different schemes(t = Tf) with N = 100')
plt.show()
plt.savefig('Problem2_b_N=100.png')

