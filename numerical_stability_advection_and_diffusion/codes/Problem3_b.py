import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard
from time import sleep
import spatial_discretization
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg  # sparse linear algebra
import scipy.linalg as scylinalg               # non-sparse linear algebra
import pylab as plt
from matplotlib import rc as matplotlibrc

machine_epsilon = np.finfo(float).eps


def Generate_Spatial_Operators_Nonperiodic(x_mesh, scheme, derivation_order=1):

    N = x_mesh.size
    circulating_row = np.zeros(N, )

    if scheme == "2nd-order-central":

        A = np.zeros((N, N))

        weights_row1 = spatial_discretization.Generate_Weights(x_mesh[:3], x_mesh[0], derivation_order)
        weights_row_last = spatial_discretization.Generate_Weights(x_mesh[-3:], x_mesh[-1], derivation_order)

        A[0, 0] = weights_row1[0]
        A[0, 1] = weights_row1[1]
        A[0, 2] = weights_row1[2]

        A[-1, -3] = weights_row_last[0]
        A[-1, -2] = weights_row_last[1]
        A[-1, -1] = weights_row_last[2]

        for i in xrange(1, Nx-1):
            weight = spatial_discretization.Generate_Weights(x_mesh[i-1:i+2], x_mesh[i], derivation_order)
            A[i, i] = weight[1]
            A[i, i-1] = weight[0]
            A[i, i+1] = weight[2]


    if scheme == "2nd-order-upwind":  # also called QUICK

        # generating computational molecule
        A = np.zeros((N, N))
        weights_row1 = spatial_discretization.Generate_Weights(x_mesh[:3], x_mesh[0], derivation_order)
        weights_row2 = spatial_discretization.Generate_Weights(x_mesh[:3], x_mesh[1], derivation_order)
        A[0, 0] = weights_row1[0]
        A[0, 1] = weights_row1[1]
        A[0, 2] = weights_row1[2]

        A[1, 0] = weights_row2[0]
        A[1, 1] = weights_row2[1]
        A[1, 2] = weights_row2[2]

        for i in xrange(2, Nx):
            weight = spatial_discretization.Generate_Weights(x_mesh[i-2:i+1], x_mesh[i], derivation_order)
            A[i, i] = weight[2]
            A[i, i-1] = weight[1]
            A[i, i-2] = weight[0]



    if scheme == "1st-order-upwind":
        # assuming advection velocity is positive, c>0
        # generating computational molecule
        x_stencil = x_mesh[:2]  # first two points
        x_eval = x_mesh[1]
        weights = spatial_discretization.Generate_Weights(x_stencil, x_eval, derivation_order)
        circulating_row[-1] = weights[0]
        circulating_row[0] = weights[1]
        A_circulant = scylinalg.circulant(circulating_row)
        A = A_circulant.transpose()

        weights_row1 = spatial_discretization.Generate_Weights(x_stencil, x_mesh[0], derivation_order)
        A[0, :] = 0
        A[0, 0] = weights_row1[0]
        A[0, 1] = weights_row1[1]

    return scysparse.csr_matrix(A)


times = 5

Nx =15
Lx = 1.*times
CFL = 0.7
c_x = .5*times  # (linear) convection speed
alpha = 5.*times*2   # diffusion coefficients
a = 1.*times
omega = 10000.
beta = 1.



## Time Advancement
# time_advancement = "Explicit-Euler"
time_advancement = "Crank-Nicolson"

## Advection Scheme
# advection_scheme = "1st-order-upwind"
# advection_scheme = "2nd-order-central"
advection_scheme = "2nd-order-upwind"

## Diffusion Scheme
diffusion_scheme = "2nd-order-central"  # always-second-order central


def u_initial(X):
    # np.power: First array elements raised to powers from second array, element-wise.
    return X*0  # lambda functions are better..

percent = np.linspace(0., 1., Nx)
x_mesh = Lx * percent
x_mesh[0] = x_mesh[0]-(x_mesh[1] - x_mesh[0])
x_mesh[-1] = x_mesh[-1]+(x_mesh[-1] - x_mesh[-2])
dt = 0.01 # 0.8*2*np.pi/omega
Tf = 1000*dt  # one complete cycle


Ieye = scysparse.identity(Nx)
Dx = Generate_Spatial_Operators_Nonperiodic(x_mesh, advection_scheme, derivation_order=1)
D2x2 = Generate_Spatial_Operators_Nonperiodic(x_mesh, diffusion_scheme, derivation_order=2)

# plt.spy(Dx)
# plt.show()


if time_advancement == "Explicit-Euler":
    A = Ieye
    B = Ieye-dt*c_x*Dx+dt*alpha*D2x2
if time_advancement == "Crank-Nicolson":
    adv_diff_Op = -dt*c_x*Dx+dt*alpha*D2x2 - beta*Ieye
    A = Ieye-0.5*adv_diff_Op
    B = Ieye+0.5*adv_diff_Op
A[0, :] = 0
A[0, 0] = 0.5
A[0, 1] = 0.5
B[0, :] = 0

A[Nx-1, :] = 0
dx_last = x_mesh[-1] - x_mesh[-2]
A[Nx-1, Nx-2] = -1/dx_last
A[Nx-1, Nx-1] = 1/dx_last
B[Nx-1, :] = 0
# print A[0, :]
# plt.spy(A)
# plt.show()
# print "B is", B

Q = np.zeros(Nx)
A, B = scysparse.csr_matrix(A), scysparse.csr_matrix(B)

u = u_initial(x_mesh)  # initializing solution

figwidth       = 10
figheight      = 6
lineWidth      = 4
textFontSize   = 28
gcafontSize    = 30



time = 0.
it   = 0

ymin = -1
ymax = 1.

plot_every = 50
delta = 0.

while time < Tf:

   it   += 1
   time += dt
   if delta <= np.abs(u[-1]/2.+u[-2]/2.):
       delta = np.abs(u[-1]/2.+u[-2]/2.)
   Q[0] = a*np.cos(omega*time)

   u = spysparselinalg.spsolve(A, B.dot(u)+Q)   # the first u is initial condition
#    if not bool(np.mod(it, plot_every)):  # plot every plot_every time steps  # np.mod is to compute reminder of division
#    # if time == time :  # plot every plot_every time steps  # np.mod is to compute reminder of division
#
#        plt.plot(x_mesh, u, label = "t = {0:.2f}".format(time))
#        plt.plot(x_mesh, u_initial(x_mesh), '--k', linewidth=1)
#        plt.grid('on', which='both')
#        plt.xlabel(r"$x$", fontsize=textFontSize)
#        plt.ylabel(r"$u(x,t)$", fontsize=textFontSize, rotation=90)
# #
#
# plt.grid('on', which='both')
# plt.xlabel(r"$x$", fontsize=textFontSize)
# plt.ylabel(r"$u(x,t)$", fontsize=textFontSize, rotation=90)
# # plt.ylim([-0.002, 0.002])
# plt.title('Increasing a')
# # plt.savefig("Problem3_a_increase a")
# # plt.show()
# print delta

delta_list = [0.0417756923705, 0.0835051384741, 0.1285317427028, 0.1607610562719, 0.2018912609195]
Lx_list = [1, 2, 3, 4, 5]
plt.plot(Lx_list, delta_list, '*-', label = 'Doundary layer thickness')
plt.title('Doundary layer thickness vs Lx')
plt.xlabel('Lx')
plt.ylabel('Boundary layer thickness')
plt.savefig('Problem3_b')
