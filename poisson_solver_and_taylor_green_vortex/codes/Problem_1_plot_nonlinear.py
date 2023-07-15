import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard
import time
import spatial_operators
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg  # sparse linear algebra
import scipy.linalg as scylinalg               # non-sparse linear algebra
import pylab as plt
from matplotlib import rc as matplotlibrc
import time # has the equivalent of tic/toc
from gauss_seidel import gs_sor
from spatial_operators import create_Derivative_operator
from gauss_seidel import gs_sor_nonlinear
machine_epsilon = np.finfo(float).eps

Re_list = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]
iters_list = []
time_list = []
for i in range(len(Re_list)):
    start_time = time.time()
    Nxc = 100
    Nyc = 100  # N * N * N * N * 8 /1024/1024/1024
    Np = Nxc*Nyc
    Lx = 1.
    Ly = 1.
    Re = Re_list[i]
    n = 16.

    #########################################
    ######## Preprocessing Stage ############

    xsi_u = np.linspace(0., 1.0, Nxc+1)
    xsi_v = xsi_u
    # uniform grid
    xu = xsi_u*Lx    # x location of the u velocity component
    yv = xsi_v*Ly

    dxu0 = np.diff(xu)[0]
    dxuL = np.diff(xu)[-1]
    xu = np.concatenate([[xu[0]-dxu0], xu, [xu[-1]+dxuL]])

    dyv0 = np.diff(yv)[0]
    dyvL = np.diff(yv)[-1]
    yv = np.concatenate([[yv[0]-dyv0], yv, [yv[-1]+dyvL]])

    dxc = np.diff(xu)  # pressure-cells spacings
    dyc = np.diff(yv)

    xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells
    yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells

    [Xc, Yc] = np.meshgrid(xc, yc)     # these arrays are with ghost cells
    [Dxc, Dyc] = np.meshgrid(dxc, dyc)   # these arrays are with ghost cells

    pressureCells_Mask = np.zeros(Xc.shape)
    pressureCells_Mask[1:-1, 1:-1] = True

    Np = len(np.where(pressureCells_Mask == True)[0])

    def f(x, y, Re, n):
        return np.sin(2.*np.pi*n*x)*np.sin(2.*np.pi*n*y)*2.*np.pi*n*np.sin(2.*np.pi*n*y)*np.cos(2.*np.pi*n*x) + 8./Re*np.pi**2.*n**2.*np.sin(2.*np.pi*n*x)*np.sin(2.*np.pi*n*y)

    def get_q(xc, yc, Re, n):

        xc_true = xc[1: -1]
        yc_true = yc[1: -1]
        q = np.zeros(len(yc_true) * len(xc_true))

        for ii in range(len(yc_true)):
            for jj in range(len(xc_true)):
                q[jj + ii * len(xc_true)] = f(xc[ii], yc[jj], Re, n)
        return q

    DivGrad = spatial_operators.create_DivGrad_operator(Dxc, Dyc, Xc, Yc, pressureCells_Mask, boundary_conditions="Homogeneous Dirichlet")
    Derivative = create_Derivative_operator(Dxc, Xc, pressureCells_Mask, boundary_conditions="Homogeneous Dirichlet")

    phi, iters = gs_sor_nonlinear(Derivative,  1. / Re * DivGrad, get_q(xc, yc, Re, n), w=1.3)
    iters_list.append(iters)
    end_time = time.time()
    time_list.append(end_time-start_time)

plt.plot(Re_list, iters_list)
plt.xlabel('Reynolds Number Re')
plt.ylabel('The number of iterations')
plt.title('The number of iterations vs Re of nonlinear equation with N = 130 and n = 16')
plt.savefig('The number of iterations vs Reynolds Number Re of nonlinear equation(a) with N = 130 and n = 16')
plt.close()

plt.plot(Re_list, time_list)
plt.xlabel('Reynolds Number Re')
plt.ylabel('Total running time')
plt.title('Running time vs Re of nonlinear equation with N = 130 and n = 16')
plt.savefig('Running time vs Reynolds Number Re of nonlinear equation(a) with N = 130 and n = 16')
plt.close()

