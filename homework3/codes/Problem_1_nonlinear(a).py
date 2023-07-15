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


Nxc = 50
Nyc = 50  # N * N * N * N * 8 /1024/1024/1024
Np = Nxc*Nyc
Lx = 1.
Ly = 1.
# Lx = 1.
# Ly = 1.
Re = 1.
n = 2.

#########################################
######## Preprocessing Stage ############

# You might have to include ghost-cells here
# Depending on your application

# define grid for u and v velocity components first
# and then define pressure cells locations
xsi_u = np.linspace(0., 1.0, Nxc+1)
xsi_v = xsi_u
# uniform grid
xu = xsi_u*Lx    # x location of the u velocity component
yv = xsi_v*Ly

# (non-sense non-uniform grid)
#xu = (xsi_u**4.)*Lx # can be replaced by non-uniform grid
#yv = (xsi_v**4.)*Ly # can be replaced by non-uniform grid

# creating ghost cells
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

# note that indexing is Xc[j_y,i_x] or Xc[j,i]
[Xc, Yc] = np.meshgrid(xc, yc)     # these arrays are with ghost cells
[Dxc, Dyc] = np.meshgrid(dxc, dyc)   # these arrays are with ghost cells

# Pre-allocated at all False = no fluid points
pressureCells_Mask = np.zeros(Xc.shape)
pressureCells_Mask[1:-1, 1:-1] = True

# number of actual pressure cells
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



# a more advanced option is to separately create the divergence and gradient operators

DivGrad = spatial_operators.create_DivGrad_operator(Dxc, Dyc, Xc, Yc, pressureCells_Mask, boundary_conditions="Homogeneous Dirichlet")
# print  scysparse.issparse(DivGrad_dir)
Derivative = create_Derivative_operator(Dxc, Xc, pressureCells_Mask, boundary_conditions="Homogeneous Dirichlet")

phi = gs_sor_nonlinear(Derivative,  1. / Re * DivGrad, get_q(xc, xc, Re, n), w=1.3)[0]
# phi = gs_sor(partial_X - 1. / Re * DivGrad_dir, get_q(xc, yc, Re, n), w=1.3)[0]
print phi

# consider pre-factorization with LU decomposition
# will speed up Poisson solver by x10

# pouring flattened solution back in 2D array
Phi = np.zeros((Nyc+2, Nxc+2))*np.nan # remember ghost cells
# keyboard()
Phi[np.where(pressureCells_Mask==True)] = phi

# Figure settings
matplotlibrc('text.latex', preamble='\usepackage{color}')
matplotlibrc('text',usetex=True)
matplotlibrc('font', family='serif')

figwidth       = 10
figheight      = 6
lineWidth      = 4
textFontSize   = 28
gcafontSize    = 30

# Plot solution
fig = plt.figure(0, figsize=(figwidth,figheight))
ax   = fig.add_axes([0.15,0.15,0.8,0.8])
plt.axes(ax)
plt.contourf(Xc, Yc, Phi)
plt.colorbar()

ax.grid('on',which='both')
plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
ax.set_xlabel(r"$x$",fontsize=textFontSize)
ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
plt.axis("tight")
plt.axis("equal")

plt.show()