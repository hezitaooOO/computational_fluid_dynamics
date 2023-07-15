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
from gauss_seidel import gauss_seidel_SOR

def f(x, y, Re, n):
    return 2.*np.pi*n*np.sin(2.*np.pi*n*y)*np.cos(2.*np.pi*n*x) + 8./Re*np.pi**2*n**2*np.sin(2.*np.pi*n*x)*np.sin(2.*np.pi*n*y)


machine_epsilon = np.finfo(float).eps

#########################################
############### User Input ##############

# number of (pressure) cells = mass conservation cells
Nxc  = 20        # number of points in x domain
Nyc  = 20         # number of points in y domain
Np   = Nxc*Nyc    # total number of points
Lx   = 2.*np.pi   # length of x domain(defined by users)
Ly   = 2.*np.pi   # length of y domain(defined by users)
Re = 1.
n = 8.
omega = 1.1
#########################################
######## Preprocessing Stage ############
xsi_u = np.linspace(0., 1.0, Nxc+1)
xsi_v = xsi_u

xu = xsi_u*Lx  # x_mesh
yv = xsi_v*Ly  # y_mesh

# creating ghost cells
dxu0 = np.diff(xu)[0]
dxuL = np.diff(xu)[-1]
xu = np.concatenate([[xu[0]-dxu0], xu, [xu[-1]+dxuL]])  # x_mesh with ghost point

dyv0 = np.diff(yv)[0]
dyvL = np.diff(yv)[-1]
yv = np.concatenate([[yv[0]-dyv0], yv, [yv[-1]+dyvL]])  # y_mesh with ghost point, shape is Nxc + 3

dxc = np.diff(xu)  # pressure-cells spacings
dyc = np.diff(yv)

xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells, shape is Nxc + 2
yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells
print xc.shape
# note that indexing is Xc[j_y,i_x] or Xc[j,i]
[Xc,Yc]   = np.meshgrid(xc,yc)     # these arrays are with ghost cells
[Dxc,Dyc] = np.meshgrid(dxc,dyc)   # these arrays are with ghost cells


# Pre-allocated at all False = no fluid points
pressureCells_Mask = np.zeros(Xc.shape)
pressureCells_Mask[1:-1, 1:-1] = True  # not included ghost cells

Np = len(np.where(pressureCells_Mask == True)[0])  # np.where(pressureCells_Mask == True) is the array of index where value of Mask is True
q  = np.ones(Np,)  # q is left-sight vector; this should be modified based on different BCs(book p747)

true_cells_x = xc[1:-1]
true_cells_y = yc[1:-1]
for i in range(Nxc):
    for j in range(Nxc):
        q[i*Nxc + j] = f(true_cells_x[i], true_cells_y[j], Re, n)



DivGrad = spatial_operators.create_DivGrad_operator(Dxc, Dyc, Xc, Yc, pressureCells_Mask, boundary_conditions="Homogeneous Dirichlet").todense()
Derivative = spatial_operators.create_Derivative_operator(Dxc, Dyc, Xc, Yc, pressureCells_Mask, boundary_conditions="Homogeneous Dirichlet").todense()


# phi = gauss_seidel_SOR(Derivative - 1./Re*DivGrad, q, omega=1.3) # solving nabla*phi = 1
A = Derivative - 1./Re*DivGrad
tol = 10**(-10)
n = len(q)
x0 = np.zeros(n, )
x = x0
norm_0 = np.linalg.norm(A.dot(x0)-q)
x_old = x0
index = 0
iters_list = []
norm_k_list = []
while True:
    print index
    index += 1

    for j in range(n):
        x[j] = (1-omega)*x_old[j] + (omega/A[j, j])*(q[j] - A[j,:].dot(x) + A[j, j]*x[j])
    x_old = x
    norm_k = np.linalg.norm(A.dot(x) - q)
    print norm_k/norm_0
    norm_k_list.append(norm_k)
    iters_list.append(index)

    if norm_k/norm_0 < tol:
        break
phi = x

Phi = np.zeros((Nyc+2, Nxc+2))*np.nan # remember ghost cells(two in both direction)
Phi[np.where(pressureCells_Mask == True)] = phi
print norm_k_list
plt.plot(iters_list, norm_k_list)
plt.yscale('log')
plt.show()


# figwidth       = 10
# figheight      = 6
# lineWidth      = 4
# textFontSize   = 28
# gcafontSize    = 30
#
# # Plot solution
# fig = plt.figure(0, figsize=(figwidth,figheight))
# ax   = fig.add_axes([0.15,0.15,0.8,0.8])
# plt.axes(ax)
# plt.contourf(Xc, Yc, Phi)
# plt.colorbar()
#
# ax.grid('on',which='both')
# plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
# plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
# ax.set_xlabel(r"$x$",fontsize=textFontSize)
# ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
# plt.axis("tight")
# plt.axis("equal")
# plt.savefig('a.png')
