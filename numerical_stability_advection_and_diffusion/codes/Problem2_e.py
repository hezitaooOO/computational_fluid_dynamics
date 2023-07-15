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

time_advancement = "Explicit-Euler"
#time_advancement = "Crank-Nicolson"

## Advection Scheme
# advection_scheme = "1st-order-upwind"
# advection_scheme = "2nd-order-central"
advection_scheme = "2nd-order-upwind"

## Diffusion Scheme
diffusion_scheme = "2nd-order-central"  # always-second-order central

Lx = 1
c_x = 10
alpha = 1


def Generate_Spectral_Radius(dt, dx):


    Nx = Lx/dx
    xx = np.linspace(0., Lx, Nx + 1)
    x_mesh = 0.5 * (xx[:-1] + xx[1:])
    Ieye = scysparse.identity(Nx)
    Dx = spatial_discretization.Generate_Spatial_Operators(x_mesh, advection_scheme, derivation_order=1)
    D2x2 = spatial_discretization.Generate_Spatial_Operators(x_mesh, diffusion_scheme, derivation_order=2)

    if time_advancement == "Explicit-Euler":
        A = Ieye
        B = Ieye - dt * c_x * Dx + dt * alpha * D2x2
    if time_advancement == "Crank-Nicolson":
        adv_diff_Op = -dt * c_x * Dx + dt * alpha * D2x2
        A = Ieye - 0.5 * adv_diff_Op
        B = Ieye + 0.5 * adv_diff_Op

    A = scysparse.csc_matrix(A)
    B = scysparse.csr_matrix(B)

    T = (scylinalg.inv(A.todense())).dot(B.todense())  # T = A^{-1}*B
    lambdas, _ = scylinalg.eig(T)
    spec_radius = max(abs(lambdas))
    return spec_radius

fig_folder = "./"
fig_name = "Problem2_e_2nd_central.png"

# This defines points in the x and y directions
C_c = np.linspace(0.1, 2., 10)
C_alpha = np.linspace(0.1, 5., 10)
n = len(C_c)
eig = np.zeros((n,n))
# print "C_c", C_c[1]
# print C_alpha[2]
# print C_c[1]*alpha/(C_alpha[2]*c_x)

dt = C_c**2.*alpha/(C_alpha*c_x**2.)
dx = C_c*alpha/(C_alpha*c_x)
print "C_c is ", C_c
print "C_alpha is ", C_alpha
print "dt is ", dt
print "dx is ", dx



for i in xrange(n):

    for j in xrange(n):
        print "for i = ", i, ", j = ",j
        eig[i][j] = Generate_Spectral_Radius(dt[i], dx[j])

print eig

# This creates a 2D grid of size (41 x 41)
# GRID = np.meshgrid(C_c, C_alpha)
# print np.shape(GRID)


# This defines points in the x and y directions
xvals = C_c
yvals = C_alpha

# This creates a 2D grid of size (41 x 41)
XX, YY = np.meshgrid(xvals, yvals)
ZZ = eig

levels = np.linspace(0, 2, 10)

# We create a figure object
ff = plt.figure(0, figsize=(8,8))
# add an axis to the figure object
ax = ff.add_subplot(111)
# add a FLOODED (the 'f' at the end of contour) contour to the axis
cc = ax.contourf(XX,YY,ZZ, levels = levels)

# Set labels and title for our primary axis containing said contour
ax.set_xlabel(r'C_c')
ax.set_ylabel(r'C_alpha')
ax.set_title(r'Spectral radius')

# squeeze the main subplot by a little to make space for a colorbar
ff.subplots_adjust(right=0.9)
# add a new subplot (axis) to the figure using ([xstart, ystart, xspan, yspan]) as percent of figure size
caxb = ff.add_axes([0.91, 0.1, 0.05, 0.8])

# generate a colorbar for contour "cc", to be plotted on axis caxb (created above)
cb = ff.colorbar(cc, cax=caxb, orientation='vertical')
# set a label for the colorbar
cb.ax.set_ylabel(r'Levels')

# save figure to disk
ff.savefig('Problem2_e_2nd_upwind_advection.png', bbox_inches='tight')

# This can be used to output the figure to screen during program execution
# BE CAREFUL: using ff.show() flushes the buffer, meaning your plots are lost and have to be re-done if you
#             wish to save them to disk as well. HOwever, saving to disk first does not flush the buffer and
#             you can first save a figure to disk and THEN use ff.show() to view it right away.
#ff.show()

# if ff.show() is the last command before plt.close, it doesn't stick around. the raw_input forces it to wait
# until we manually choose to exit.
#_ = raw_input('Press any key to end program...')

# this will clear all figures currently being plotted.
# If you don't do this when plotting multiple figures one after the other, and the buffer is not flushed out (and its not when simply saving the figure to disk, as you will in your homework codes) then the 2nd figure will contain the first AND second plots. It is therefore necessary to use plt.close('all') to clear all figures before you start plotting another one.
plt.close('all')

