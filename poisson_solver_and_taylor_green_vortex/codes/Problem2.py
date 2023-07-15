import os
import sys
import numpy as np
from symbol import *
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
from spatial_operators import create_DivGrad_operator



def u_analytical(x, y, t):
    return -np.exp(-2.*t)*np.cos(x)*np.sin(y)

def v_analytical(x, y, t):
    return np.exp(-2.*t)*np.sin(x)*np.cos(y)

def p_analytical(x, y, t):
    return -np.exp(-4.*t)/4.*(np.cos(2.*x)+np.cos(2.*y))



def u_partial_x(x, y, t):
    return np.exp(-2.*t)*np.sin(x)*np.sin(y)

def u_partial_y(x, y, t):
    return -np.exp(-2.*t)*np.cos(x)*np.cos(y)

def v_partial_x(x, y, t):
    return np.exp(-2. * t) * np.cos(x) * np.cos(y)

def v_partial_y(x, y, t):
    return -np.exp(-2. * t) * np.sin(x) * np.sin(y)




def get_first_derivative_upwind_over_x(field, dx, dy):
    field_E = np.roll(field, -1, axis=1)  # roll left
    field_C = field
    field_derivative = (field_E - field_C)/dx
    return  field_derivative

def get_first_derivative_upwind_over_y(field, dx, dy):
    field_N = np.roll(field, 1, axis=0) # roll down
    field_C = field
    field_derivative = (field_N - field_C)/dy
    return  field_derivative




def get_analytical_convective_operator_u(x, y, t):
    return -np.sin(2.*x)*np.exp(-4.*t)*(np.sin(y)**2.+1./2.*np.cos(2.*y))

def get_analytical_convective_operator_v(x, y, t):
    return -np.sin(2.*y)*np.exp(-4.*t)*(np.sin(x)**2.+1./2.*np.cos(2.*x))

def get_viscous_operator(grid, dx, dy):

    delta = dx

    grid_E = np.roll(grid, -1, axis=1) # roll left
    grid_W = np.roll(grid, 1, axis=1) # roll right
    grid_N = np.roll(grid, 1, axis=0) # roll down
    grid_S = np.roll(grid, -1, axis=0) # roll up
    grid_C = grid

    AA = (grid_E+grid_W-2.*grid_C)/(dx**2.)
    BB = (grid_N+grid_S-2.*grid_C)/(dy**2.)

    return AA+BB

def get_convective_operator_u(u, v, dx, dy):


    v_se = np.delete(v, (-1), axis=0) # delete the last row
    v_se = np.append(v_se, (v_se[:, 0]).reshape(len(v_se), 1), axis=1)  # copy first column to last column

    v_ne = np.delete(v, (0), axis=0)  # delete the first row
    v_ne = np.append(v_ne, (v_ne[:, 0]).reshape(len(v_ne), 1), axis=1)  # copy first column to last column

    v_nw = np.delete(v, (0), axis=0)  # delete the first row
    v_nw = np.concatenate(((v_nw[:,-1]).reshape(len(v_nw), 1), v_nw), axis=1)  # copy last column to first column

    v_sw = np.delete(v, (-1), axis=0) # delete the last row
    v_sw = np.concatenate(((v_sw[:,-1]).reshape(len(v_sw), 1), v_sw), axis=1)  # copy last column to first column

    u_E = np.roll(u, -1, axis=1)  # roll left
    u_W = np.roll(u, 1, axis=1)  # roll right
    u_N = np.roll(u, 1, axis=0)  # roll down
    u_S = np.roll(u, -1, axis=0)  # roll up
    u_C = u

    element_1 = ((u_E+u_C)/2.)**2.
    element_2 = ((u_W+u_C)/2.)**2.
    element_3 = ((u_C+u_N)/2.) * ((v_ne + v_nw)/2.)
    element_4 = ((u_C+u_S)/2.) * ((v_se + v_sw)/2.)

    A = -element_1/dx + element_2/dx - element_3/dy + element_4/dy
    return A

def get_convective_operator_v(u, v, dx, dy):

    v_E = np.roll(v, -1, axis=1)  # roll left
    v_W = np.roll(v, 1, axis=1)  # roll right
    v_N = np.roll(v, 1, axis=0)  # roll down
    v_S = np.roll(v, -1, axis=0)  # roll up
    v_C = v

    u_se = np.delete(u, (0), axis=1) # delete the first column
    u_se = np.concatenate(((u_se[-1,:]).reshape(1, len(u_se)), u_se), axis=0)  # copy last row to first row

    u_ne = np.delete(u, (0), axis=1) # delete the first column
    u_ne = np.append(u_ne, (u_ne[0, :]).reshape(1, len(u_ne)), axis=0)  # copy first row to last row

    u_nw = np.delete(u, (-1), axis=1) # delete the last column
    u_nw = np.append(u_nw, (u_nw[0, :]).reshape(1, len(u_nw)), axis=0)  # copy first row to last row

    u_sw = np.delete(u, (-1), axis=1) # delete the last column
    u_sw = np.concatenate(((u_sw[-1,:]).reshape(1, len(u_sw)), u_sw), axis=0)  # copy last row to first row

    element_1 = ((u_ne+u_se)/2.) * ((v_C+v_E)/2.)
    element_2 = ((u_nw+u_sw)/2.) * ((v_C+v_W)/2.)
    element_3 = ((v_C+v_N)/2.)**2.
    element_4 = ((v_C+v_S)/2.)**2.

    A = -element_1/dx + element_2/dx - element_3/dy + element_4/dy
    return A




Nxc  = 5        # number of points in x domain
Nyc  = 5         # number of points in y domain
Np   = Nxc*Nyc    # total number of points
Lx   = 2.*np.pi   # length of x domain(defined by users)
Ly   = 2.*np.pi   # length of y domain(defined by users)
dt = 1e-3
t_end = 1.
t = 0.
dx = Lx/Nxc
dy = Ly/Nyc
print "dx is: \n", dx
######################################################################
######## Step 1: solve Helmmolte Equation(VEL prediction) ############

ux = np.linspace(0., Lx, Nxc+1) # grid of u is a N by N+1 matrix
uy = 0.5*(ux[:-1]+ux[1:])
[UX, UY] = np.meshgrid(ux, uy)

vx = uy      # grid of v is a N+1 by N matrix
vy = ux
[VX, VY] = np.meshgrid(vx, vy)

px = vx       # grid of p is a N by N matrix
py = vx
[PX, PY]= np.meshgrid(px, py)

u = u_analytical(UX, UY, t = 0.)   # initial value of u, v, p when t = 0
v = v_analytical(VX, VY, t = 0.)
p = p_analytical(PX, PY, t = 0.)

# print "u is: ", u
# print "v is: ", v
# # print "viscous term is: \n", get_viscous_operator(u,dx,dy)
print "convective for u is: \n", get_convective_operator_u(u,v,dx,dy)
print "convective for v is: \n", get_convective_operator_v(u,v,dx,dy)
# # print VX
# # print VY
print "analytical convective for u is: \n", get_analytical_convective_operator_u(UX, UY, t = 0)
print "analytical convective for v is: \n", get_analytical_convective_operator_v(VX, VY, t = 0)

# print "analytical convective for u is: \n", u_initial(UX[:,1:], UY[:,1:], t)*u_partial_x(UX[:,1:], UY[:,1:], t) + v_initial(UX[:,1:], UY[:,1:], t)*u_partial_y(UX[:,1:], UY[:,1:], t)
# print "analytical convective for v is: \n", u_initial(VX[1:,:], VY[1:,:], t)*v_partial_x(VX[1:,:], VY[1:,:], t) + v_initial(VX[1:,:], VY[1:,:], t)*v_partial_y(VX[1:,:], VY[1:,:], t)

R_vis_u = get_viscous_operator(u, dx, dy)   # viscous term of u,v
R_vis_v = get_viscous_operator(v, dx, dy)
R_conv_u = get_convective_operator_u(u, v, dx, dy)  # convective term of u, v
R_conv_v = get_convective_operator_v(u, v, dx, dy)
u_star = u+dt*(R_vis_u + R_conv_u)    # u_star and v_star
v_star = v+dt*(R_vis_v + R_conv_v)


##########################################################################################################################################
######################construct Laplasce operaror for P(copy from HW3 sample, xy and yv are the same of px and py)########################
xsi_u = np.linspace(0., 1.0, Nxc+1)
xsi_v = xsi_u
# uniform grid
xu = xsi_u*Lx  # x_mesh
yv = xsi_v*Ly  # y_mesh
# (non-sense non-uniform grid)
#xu = (xsi_u**4.)*Lx # can be replaced by non-uniform grid
#yv = (xsi_v**4.)*Ly # can be replaced by non-uniform grid

# creating ghost cells
dxu0 = np.diff(xu)[0]
dxuL = np.diff(xu)[-1]
xu = np.concatenate([[xu[0]-dxu0], xu, [xu[-1]+dxuL]])  # x_mesh with ghost point

dyv0 = np.diff(yv)[0]
dyvL = np.diff(yv)[-1]
yv = np.concatenate([[yv[0]-dyv0], yv, [yv[-1]+dyvL]])  # y_mesh with ghost point

dxc = np.diff(xu)  # pressure-cells spacings
dyc = np.diff(yv)

xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells
yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells
# note that indexing is Xc[j_y,i_x] or Xc[j,i]
[Xc,Yc]   = np.meshgrid(xc,yc)     # these arrays are with ghost cells
[Dxc,Dyc] = np.meshgrid(dxc,dyc)   # these arrays are with ghost cells
pressureCells_Mask = np.zeros(Xc.shape)
pressureCells_Mask[1:-1, 1:-1] = True  # not included ghost cells
DivGrad = create_DivGrad_operator(Dxc, Dyc, Xc, Yc, pressureCells_Mask)
######################  construct Laplasce operator for P done      ################
####################################################################################



###################################################################################################
##################       Step 2: solve poisson equation for pressure        ########################

u_star_over_x = get_first_derivative_upwind_over_x(u_star, dx, dy)
v_star_over_y = get_first_derivative_upwind_over_y(v_star, dx, dy)
RHS =  u_star_over_x[:, :-1] + v_star_over_y[:-1, :]  #RHS is righthand-sided vector for poisson equation for pressure
RHS = RHS/dt
RHS = RHS.flatten()   # convert RHS from matrix to vector
p_next = spysparselinalg.spsolve(DivGrad, RHS)  # p_next is a vector, have to convert to matrix to move on



######################                    step 2 done       ########################
####################################################################################


###################################################################################################
##################       Step 3: obtain new velocity u_n+1 and v_n+1       ########################

p_next = p_next.reshape((Nxc,Nxc))
print p_next.reshape((Nxc,Nxc)).shape

p_over_x = get_first_derivative_upwind_over_x(p_next, dx, dy)
p_over_x = np.concatenate(((p_over_x[:,-1]).reshape(len(p_over_x), 1), p_over_x), axis=1) # # copy last column to first column

p_over_y = get_first_derivative_upwind_over_y(p_next, dx, dy)
p_over_y = np.concatenate(((p_over_y[-1,:]).reshape(1, len(p_over_y)), p_over_y), axis=0)  # copy last row to first row

u_next = u_star - p_over_x/dt
v_next = v_star - p_over_y/dt

print u_next
