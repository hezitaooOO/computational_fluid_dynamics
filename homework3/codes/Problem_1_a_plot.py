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
from gauss_seidel import gs_sor_nonlinear
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


N_list = [10, 20, 30, 40, 50]
k_list_n1= []
k_list_n4= []
k_list_n8= []



for i in range(len(N_list)):
    Nxc = N_list[i]
    Nyc = N_list[i]
    Np = Nxc*Nyc
    Lx = 1.
    Ly = 1.
    Re = 1.
    n = 1.

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

    DivGrad_dir = spatial_operators.create_DivGrad_operator(Dxc, Dyc, Xc, Yc, pressureCells_Mask, boundary_conditions="Homogeneous Dirichlet")
    Derivative = create_Derivative_operator(Dxc, Xc, pressureCells_Mask, boundary_conditions="Homogeneous Dirichlet")

    phi, k= gs_sor(Derivative - 1. / Re * DivGrad_dir, get_q(xc, yc, Re, n), w=1.3)
    k_list_n1.append(k)
    Phi = np.zeros((Nyc+2, Nxc+2))*np.nan # remember ghost cells
    Phi[np.where(pressureCells_Mask==True)] = phi



for i in range(len(N_list)):
    
    Nxc = N_list[i]
    Nyc = N_list[i]
    Np = Nxc*Nyc
    Lx = 1.
    Ly = 1.
    Re = 1.
    n = 4.

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

    DivGrad_dir = spatial_operators.create_DivGrad_operator(Dxc, Dyc, Xc, Yc, pressureCells_Mask, boundary_conditions="Homogeneous Dirichlet")
    partial_X = create_Derivative_operator(Dxc, Xc, pressureCells_Mask, boundary_conditions="Homogeneous Dirichlet")

    phi, k= gs_sor(partial_X - 1. / Re * DivGrad_dir, get_q(xc, yc, Re, n), w=1.3)
    k_list_n4.append(k)
    Phi = np.zeros((Nyc+2, Nxc+2))*np.nan # remember ghost cells
    Phi[np.where(pressureCells_Mask==True)] = phi




for i in range(len(N_list)):
    Nxc = N_list[i]
    Nyc = N_list[i]
    Np = Nxc*Nyc
    Lx = 1.
    Ly = 1.
    Re = 1.
    n = 8.

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

    DivGrad_dir = spatial_operators.create_DivGrad_operator(Dxc, Dyc, Xc, Yc, pressureCells_Mask, boundary_conditions="Homogeneous Dirichlet")
    partial_X = create_Derivative_operator(Dxc, Xc, pressureCells_Mask, boundary_conditions="Homogeneous Dirichlet")

    phi, k= gs_sor(partial_X - 1. / Re * DivGrad_dir, get_q(xc, yc, Re, n), w=1.3)
    k_list_n8.append(k)
    Phi = np.zeros((Nyc+2, Nxc+2))*np.nan # remember ghost cells
    Phi[np.where(pressureCells_Mask==True)] = phi

grid_size = []
for j in range(len(N_list)):
    grid_size.append(1. / N_list[j])

order1 = [1./x**1. for x in grid_size]
order2 = [1./x**2. for x in grid_size]
order3 = [1./x**3. for x in grid_size]

plt.plot(grid_size, order1, '--', label = '1st order')
plt.plot(grid_size, order2, '--', label = '2nd order')

plt.plot(grid_size, k_list_n1, label = 'wave number n = 1')
plt.plot(grid_size, k_list_n4, label = 'wave number n = 4')
plt.plot(grid_size, k_list_n8, label = 'wave number n = 8')
plt.title('Number of iterations k versus girdsize N')
plt.xlabel('Gridsize dx')
plt.ylabel('Number of iterations k')
plt.yscale('log')
plt.xscale('log')
plt.xlim((0.02, 0.1))

plt.legend(loc = 'best')
plt.savefig('Number of iterations k versus girdsize N')
plt.show()


