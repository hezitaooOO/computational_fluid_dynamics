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
import datetime
from matplotlib.pyplot import streamplot

time_start = datetime.datetime.now()
N = int(15)
L = 1.0

u =     np.zeros((N+1, N))
u_new = np.zeros((N+1, N))
u_c =   np.zeros((N,N))

v =     np.zeros((N, N+1))
v_new = np.zeros((N, N+1))
v_c =   np.zeros((N,N))

p =     np.zeros((N+1, N+1))
p_new = np.zeros((N+1, N+1))
p_c =   np.zeros((N,N))

m = np.zeros((N+1, N+1))
w = np.zeros((N, N))

dx = L/(N-1)
dy = L/(N-1)
dt = 0.001
delta = 4.5
Re = 1000.0
U = 1.0
viscosity = U*L/Re
step = int(1)
tol = 10e-3
error = 1.0
# initialize u (all zeros except for top 2 layers)
u[-2:, :] = 1.0
#initialize v (all zeros)
v[:] = 0.0
#initialize p (all ones)
p[:] = 1.0


#Solves u-momentum
while error > tol:

    for i in range(1, N):
        for j in range(1, N-1):
            u_new[i, j] = u[i, j] - dt*(  (u[i, j+1]**2.0 - u[i, j-1]**2.0)/2.0/dx + 0.25*( (u[i, j]+u[i+1, j])*(v[i, j]+v[i, j+1]) - (u[i, j]+u[i-1, j])*(v[i-1, j+1]+v[i-1, j]) )/dy )\
                                            - dt/dx*(p[i, j+1] - p[i, j]) \
                                            + dt*viscosity*( (u[i, j+1] - 2.0*u[i, j] + u[i, j-1])/dx/dx + (u[i+1, j] - 2.0*u[i, j] + u[i-1, j])/dy/dy )

    # Boundary conditions for u
    u_new[1:-1, 0] = 0.0
    u_new[1:-1, -1] = 0.0
    u_new[0, :] = -u_new[1, :]
    u_new[-1, :] = 2.0 - u_new[-2, :]


    #Solves v-momentum


    for i in range(1,N-1):
        for j in range(1, N):
            v_new[i, j] = v[i, j] - dt* ( (0.25*((u[i, j]+u[i+1, j])*(v[i, j]+v[i, j+1]) - (u[i, j-1]+u[i+1, j-1])*(v[i, j]+v[i, j-1])))/dx
                                         + (v[i+1, j]**2.0-v[i-1, j]**2.0)/2.0/dy) \
                                         - dt/dy*(p[i+1, j]-p[i, j]) \
                                         + dt*viscosity*((v[i, j+1]-2.0*v[i, j]+v[i, j-1])/dx/dx + (v[i+1, j]-2.0*v[i, j]+v[i-1, j])/dy/dy)

    # Boundary conditions for v

    v_new[1:-1, 0] = -v_new[1:-1, 1]
    v_new[1:-1, -1] = -v_new[1:-1, -2]
    v_new[0, :] = 0.0
    v_new[-1, :] = 0.0


    # Solve continuity equation

    #pn[i][j] = p[i][j]-dt*delta*(  ( un[i][j]-un[i-1][j] )/dx + ( vn[i][j]-vn[i][j-1] ) /dy  );

    for i in range(1, N):
        for j in range(1, N):
            p_new[i, j] = p[i, j] - dt*delta*( (u_new[i, j] - u_new[i, j-1])/dx + (v_new[i, j] - v_new[i-1, j])/dy )


    # Boundary condition for p
    p_new[0, 1:-1] = p_new[1, 1:-1]
    p_new[-1, 1:-1] = p_new[-2, 1:-1]
    p_new[:, 0] = p_new[:, 1]
    p_new[:, -1] = p_new[:, -2]

    # calculate error
    # m[i][j] = (  ( un[i][j]-un[i-1][j] )/dx + ( vn[i][j]-vn[i][j-1] )/dy  );
    # error = error + fabs(m[i][j]);

    error = 0.0
    for i in range (1, N):
        for j in range(1, N):
            m[i, j] = (u_new[i, j] - u_new[i, j-1])/dx + (v_new[i, j] - v_new[i-1, j])/dy
            error = error+abs(m[i, j])


    # print step

    if (step % 1000 ==1):
        print "Error is {} for step {}".format(error, step)

    # update solution
    u = u_new
    v = v_new
    p = p_new

    step += 1



for i in range(0, N):
    for j in range(0, N):
        u_c[i, j] = 0.5 * (u_new[i, j] + u_new[i+1, j])
        v_c[i, j] = 0.5 * (v_new[i, j] + v_new[i, j+1])
        p_c[i, j] = 0.25 * (p_new[i, j] + p_new[i, j+1] + p_new[i+1, j] + p_new[i+1, j+1])

# calculate vorticity

for i in range(0, N):
    for j in range(0, N):
        w[i, j] = (v_new[i, j + 1] - v_new[i, j]) / dx - (u_new[i + 1, j] - u_new[i, j]) / dy


x = np.linspace(0.0, L, N)
y = np.linspace(0.0, L, N)
X, Y = np.meshgrid(x, y)



streamplot(x, y, u_c, v_c, density= 2)
plt.savefig('streamline countour Re = 100')
plt.show()
plt.close()

########################################################
###########   calculate streamfunction   ###############
x_c = np.concatenate([[x[0]-dx],x,[x[-1]+dx]])
y_c = np.concatenate([[y[0]-dy],y,[y[-1]+dy]])
[Xc, Yc] = np.meshgrid(x_c, y_c)
pressureCells_Mask = np.zeros(Xc.shape)
pressureCells_Mask[1:-1,1:-1] = True
Dxc = np.zeros(Xc.shape)+dx
Dyc = np.zeros(Yc.shape)+dy


DivGrad = spatial_operators.create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,boundary_conditions="Homogeneous Dirichlet")

q = -w.flatten()

psi = spysparselinalg.spsolve(DivGrad, q)
psi_c = np.reshape(psi, (N, N))

levels = np.linspace(np.amin(psi_c), np.amax(psi_c), 41)
ff = plt.figure(0, figsize=(8,8))
ax = ff.add_subplot(111)
cc = ax.contourf(X,Y,psi_c, levels = levels)
ff.subplots_adjust(right=0.85)
caxb = ff.add_axes([0.86, 0.1, 0.05, 0.8])
cb = ff.colorbar(cc, cax=caxb, orientation='vertical')
plt.savefig('psi contour Re = 100')
plt.close()
########################################################

time_end = datetime.datetime.now()

print "running time is {}".format(time_end - time_start)





# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.contourf(X, Y, v_c)
# plt.savefig('v contour')
# plt.close()

levels = np.linspace(np.amin(u_c), np.amax(u_c), 41)
ff = plt.figure(0, figsize=(8,8))
ax = ff.add_subplot(111)
cc = ax.contourf(X,Y,u_c, levels = levels)
ff.subplots_adjust(right=0.85)
caxb = ff.add_axes([0.86, 0.1, 0.05, 0.8])
cb = ff.colorbar(cc, cax=caxb, orientation='vertical')
plt.savefig('u contour Re = 100')
plt.show()
plt.close()

levels = np.linspace(np.amin(v_c), np.amax(v_c), 41)
ff = plt.figure(0, figsize=(8,8))
ax = ff.add_subplot(111)
cc = ax.contourf(X,Y,v_c, levels = levels)
ff.subplots_adjust(right=0.85)
caxb = ff.add_axes([0.86, 0.1, 0.05, 0.8])
cb = ff.colorbar(cc, cax=caxb, orientation='vertical')
plt.savefig('v contour Re = 100')
plt.show()
plt.close()

levels = np.linspace(np.amin(p_c), np.amax(p_c), 41)
ff = plt.figure(0, figsize=(8,8))
ax = ff.add_subplot(111)
cc = ax.contourf(X,Y,p_c, levels = levels)
ff.subplots_adjust(right=0.85)
caxb = ff.add_axes([0.86, 0.1, 0.05, 0.8])
cb = ff.colorbar(cc, cax=caxb, orientation='vertical')
plt.savefig('p contour Re = 100')
plt.show()
plt.close()



# Data with Re = 100:
# Ghia_u = [1.00000,0.84123,0.78871,0.73722,0.68717,0.23151,0.00332,-0.13641,-0.20581,-0.21090,-0.15662,-0.10150,-0.06434,-0.04775,-0.04192,-0.03717,0.00000,]
# Ghia_v = [0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533,0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.00000]
# y_Ghia = [129,126,125,124,123,110,95,80,65,59,31,23,14,10,9,8,1]
# x_Ghia = [129,125,124,123,122,117,111,104,65,31,30,21,13,11,10,9,1]

# Data with Re = 1000:
Ghia_u = [1.00000,0.65928,0.57492,0.51117,0.46604,0.33304,0.18719,0.05702,-0.06080,-0.10648,-0.27805,-0.38289,-0.29730,-0.22220,-0.20196,-0.18109,0.00000]
Ghia_v = [0.00000,-0.21388,-0.27669,-0.33714,-0.39188,-0.51550,-0.42665,-0.31966,0.02526,0.32235,0.33075,0.37095,0.32627,0.30353,0.29012,0.27485,0.00000]
y_Ghia = [129,126,125,124,123,110,95,80,65,59,37,23,14,10,9,8,1]
x_Ghia = [129,125,124,123,122,117,111,104,65,31,30,21,13,11,10,9,1]



for i in range(len(y_Ghia)):
    y_Ghia[i] = y_Ghia[i]/129.0*L


for j in range(len(x_Ghia)):
    x_Ghia[j] = x_Ghia[j]/129.0*L

y = np.linspace(0.0, L, N)
x = np.linspace(0.0, L, N)

u_center = u_c[:, (N-1)/2+1]
v_center = v_c[(N-1)/2+1, :]

plt.figure(figsize=(12, 5))
p1 = plt.subplot(121)
p2 = plt.subplot(122)
p1.plot(Ghia_u, y_Ghia, marker='o', linestyle='--', color='r', label='Ghia u')
p1.plot(u_center, y, label='Numerical result for u')

p2.plot(x_Ghia, Ghia_v, marker='o', linestyle='--', color='r', label='Ghia v')
p2.plot(x, v_center, label='Numerical result for v')


p1.legend(loc = 'best')
p2.legend(loc = 'best')

p1.set_xlabel('u')
p1.set_ylabel('y location')
p1.set_title('u')

p2.set_ylabel('v')
p2.set_xlabel('x location')
p2.set_title('v')

plt.savefig("validation for u and v Re = 100")
plt.show()
plt.close()
