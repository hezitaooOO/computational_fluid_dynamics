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
from matplotlib import rc as matplotlibrc
import time # has the equivalent of tic/toc
from gauss_seidel import gs_sor
from spatial_operators import create_Derivative_operator
machine_epsilon = np.finfo(float).eps

def f(x, y, Re, n):
    return 2.*np.pi*n*np.sin(2.*np.pi*n*y)*np.cos(2.*np.pi*n*x) + 8./Re*np.pi**2.*n**2.*np.sin(2.*np.pi*n*x)*np.sin(2.*np.pi*n*y)

def get_q(xc, yc, Re, n):

    xc_true = xc[1: -1]
    yc_true = yc[1: -1]
    q = np.zeros(len(yc_true) * len(xc_true))

    for ii in range(len(yc_true)):
        for jj in range(len(xc_true)):
            q[jj + ii * len(xc_true)] = f(xc[ii], yc[jj], Re, n)
    return q

def gs_sor(A, b, w=1.4, tol=1.0e-7):
    rk_list = []
    index_list = []
    n = len(b)
    # A = np.squeeze(np.asarray(A))
    # b = np.squeeze(np.asarray(b))
    Temp = A.diagonal()
    D_vec = scysparse.diags(Temp, 0)
    D = scysparse.csr_matrix(D_vec)
    # D = np.diagflat(D_vec)

    U = scysparse.triu(A, k=1)
    L = scysparse.tril(A, k=-1)
    L = scysparse.csr_matrix(L)
    I = scysparse.eye(n, dtype=float)
    phi0 = np.zeros(n)
    phi0 = phi0
    phi = phi0
    r0 = np.linalg.norm(np.array(A.dot(phi0) - b))
    phi_old = phi0
    r = np.linalg.norm(np.array(A.dot(phi_old) - b))
    index = 0

    inv = spysparselinalg.splu(D + w*L)


    while r / r0 > tol:
        print r / r0
        index += 1
        print index
        A1 = inv.solve(((1 - w) * D - w * U).dot(phi_old))
        A2 = w*inv.solve(b)
        phi = A1 + A2
        r = np.linalg.norm(A.dot(phi) - b)
        index_list.append(index)
        rk_list.append(r)
        phi_old = phi

    return phi, index_list, rk_list






n_list= [1., 4., 8., 16., 32.]

iteration_list = []
rk_list = []

for i in range(len(n_list)):

    Nxc = 30
    Nyc = 30
    Np = Nxc*Nyc
    Lx = 1.
    Ly = 1.
    Re = 1.
    n = n_list[i]

    xsi_u = np.linspace(0., 1.0, Nxc+1)
    xsi_v = xsi_u
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

    DivGrad = spatial_operators.create_DivGrad_operator(Dxc, Dyc, Xc, Yc, pressureCells_Mask, boundary_conditions="Homogeneous Dirichlet")
    Derivative = create_Derivative_operator(Dxc, Xc, pressureCells_Mask, boundary_conditions="Homogeneous Dirichlet")

    phi, iteration, rk = gs_sor(Derivative - 1. / Re * DivGrad, get_q(xc, yc, Re, n), w=1.3)
    iteration_list.append(iteration)
    rk_list.append(rk)

# order1 = [(1/x) ** 1 for x in [float(i) for i in iteration_list[0]]]
# order2 = [(1/x) ** 2 for x in [float(i) for i in iteration_list[0]]]
# order3 = [(1/x) ** 3 for x in [float(i) for i in iteration_list[0]]]
# order4 = [(1/x) ** 4 for x in [float(i) for i in iteration_list[0]]]
#
# plt.loglog(iteration_list[0], order1)
# plt.loglog(iteration_list[0], order2)
# plt.plot(iteration_list[0], order3)
# plt.plot(iteration_list[0], order4)
#
plt.plot(iteration_list[0], rk_list[0], label = 'n = 1')
plt.plot(iteration_list[1], rk_list[1], label = 'n = 4')
plt.plot(iteration_list[2], rk_list[2], label = 'n = 8')
plt.plot(iteration_list[3], rk_list[3], label = 'n = 16')
plt.plot(iteration_list[4], rk_list[4], label = 'n = 32')


plt.title('Instantenous residual rk, versus the iteration number k for different wave number n')
plt.xlabel('iteration')
plt.ylabel('residual rk')
plt.yscale('log')
# plt.xlim((0,500))
# plt.ylim((0,10.**7))



plt.legend(loc = 'best')
plt.savefig('Instantenous residual rk, versus the iteration number k for different wave number n')
print rk_list[0]