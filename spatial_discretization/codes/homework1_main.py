import numpy as np
import homework1_Q1 as Q1
import matplotlib.pyplot as plt
import scipy.sparse as scysparse
import Generate_Spatial_Operators as gso
import spatial_discretization as sd  # like include .h file in C++, calls another file
import scipy.sparse.linalg as splinalg
from numpy.linalg import matrix_rank
from scipy.linalg import lu

Problem1and2a=0
Problem21=0
Problem22=1
Problem3=0

##############################################################################
#        Problem 1
##############################################################################
plt.figure(1)
Q1.plot_collocated_centered(2)  # the number is number of nodes(l+r)
plt.title("collocated centered scheme with l=r=1")
plt.savefig(("/Users/Tao/Desktop/ME614/howework/homework1/figures/collocated_centered_l=r=1.png"))

plt.figure(2)
Q1.plot_collocated_centered(4)
plt.title("collocated centered scheme with l=r=2")
plt.savefig(("/Users/Tao/Desktop/ME614/howework/homework1/figures/collocated_centered_l=r=2.png"))

plt.figure(3)
Q1.plot_collocated_centered(6)
plt.title("collocated centered scheme with l=r=3")
plt.savefig(("/Users/Tao/Desktop/ME614/howework/homework1/figures/collocated_centered_l=r=3.png"))
#
# ############################################################
plt.figure(4)
Q1.plot_collocated_biased(1)
plt.title("collocated biased scheme with l=0 r=1")
plt.savefig(("/Users/Tao/Desktop/ME614/howework/homework1/figures/collocated_biased_l=0_r=1.png"))

# plt.figure(5)
Q1.plot_collocated_biased(2)
plt.title("collocated biased scheme with l=0 r=2")
plt.savefig(("/Users/Tao/Desktop/ME614/howework/homework1/figures/collocated_biased_l=0_r=2.png"))

plt.figure(6)
Q1.plot_collocated_biased(3)
plt.title("collocated biased scheme with l=0 r=3")
plt.savefig(("/Users/Tao/Desktop/ME614/howework/homework1/figures/collocated_biased_l=0_r=3.png"))

# ############################################################

plt.figure(7)
Q1.plot_staggered_centered(2)
plt.title("staggered centered scheme with l=r=1")
plt.savefig(("/Users/Tao/Desktop/ME614/howework/homework1/figures/staggered_centered_l=r=1.png"))

plt.figure(8)
Q1.plot_staggered_centered(4)
plt.title("staggered centered scheme with l=r=2")
plt.savefig(("/Users/Tao/Desktop/ME614/howework/homework1/figures/staggered_centered_l=r=2.png"))

plt.figure(9)
Q1.plot_staggered_centered(6)
plt.title("staggered centered scheme with l=r=3")
plt.savefig(("/Users/Tao/Desktop/ME614/howework/homework1/figures/staggered_centered_l=r=3.png"))
#
#
# ############################################################
#
plt.figure(10)
Q1.plot_staggered_biased(2)
plt.title("staggered biased scheme with l=r=1")
plt.savefig(("/Users/Tao/Desktop/ME614/howework/homework1/figures/staggered_biased_l=1_r=1.png"))

Q1.plot_staggered_biased(4)
plt.title("staggered biased scheme with l=r=2")
plt.savefig(("/Users/Tao/Desktop/ME614/howework/homework1/figures/staggered_biased_l=1_r=2.png"))

plt.figure(12)
Q1.plot_staggered_biased(6)
plt.title("staggered biased scheme with l=r=3")
plt.savefig(("/Users/Tao/Desktop/ME614/howework/homework1/figures/staggered_biased_l=1_r=3.png"))
#
#
# ##############################################################################
# #                   Problem 2(a)
# ##############################################################################
N = 10
dx = 0.6/(N-1)
x_mesh = np.linspace(-0.3, 0.3, N)

S=gso.Generate_Spatial_Operators(x_mesh, "3rd-order", 1)
plt.figure(13)
plt.spy(S)
plt.title("1st derivatives operator with 3rd_order polynomial reconstruction")
plt.savefig("/Users/Tao/Desktop/ME614/howework/homework1/figures/1st derivative with 3rd polynomial.png")

S=gso.Generate_Spatial_Operators(x_mesh, "3rd-order", 3)
plt.figure(14)
plt.spy(S)
plt.title("3rd derivatives operator with 3rd-order polynomial reconstruction")
plt.savefig("/Users/Tao/Desktop/ME614/howework/homework1/figures/3rd derivative with 3rd polynomial.png")

S=gso.Generate_Spatial_Operators(x_mesh, "5th-order", 1)
plt.figure(15)
plt.spy(S)
plt.title("1st derivatives operator with 5th_order polynomial reconstruction")
plt.savefig("/Users/Tao/Desktop/ME614/howework/homework1/figures/1st derivative with 5th polynomial.png")

S=gso.Generate_Spatial_Operators(x_mesh, "5th-order", 3)
plt.figure(16)
plt.spy(S)
plt.title("3rd derivatives operator with 5th_order polynomial reconstruction")
plt.savefig("/Users/Tao/Desktop/ME614/howework/homework1/figures/3rd derivative with 5th polynomial.png")

# ##############################################################################
# #                     Problem 2(b)
# ##############################################################################
while Problem21 == 1:

    N = 10
    dx = 0.6 / (N - 1)
    print dx
    x_mesh = np.linspace(-0.3, 0.3, N)

    dx_inverse_list = list()
    RMS_list = list()
    j = 0

    while dx > 0.00001:
        j = j+1
        print j
        x_mesh = np.linspace(-0.3, 0.3, N)

        first_row_3rd = np.zeros(N)
        lastsecond_row_3rd = np.zeros(N)
        last_row_3rd = np.zeros(N)

        first_row_3rd[0] = 1
        last_row_3rd[-4:] = sd.Generate_Weights(x_mesh[-4:], x_mesh[N-1], 1)
        lastsecond_row_3rd[:4] = sd.Generate_Weights(x_mesh[0:4], x_mesh[0], 1)

        operator_3rd = gso.Generate_Spatial_Operators(x_mesh, "3rd-order", 3)
        operator_3rd[0, :] = first_row_3rd
        operator_3rd[N-2, :] = lastsecond_row_3rd
        operator_3rd[N-1, :] = last_row_3rd

        right_side = np.zeros(N)
        u_true = np.zeros(N)
        error = np.zeros(N)
        poly = np.poly1d([2, 3, 4, 5, 6]) # this is the function I defined

        for i in range(int(N)):
            u_true[i] = np.polyval(poly, x_mesh[i])

        right_side[0] = np.polyval(poly, x_mesh[0])
        right_side[N-2] = np.polyval(np.polyder(poly, 1), x_mesh[0])
        right_side[N-1] = np.polyval(np.polyder(poly, 1), x_mesh[N-1])

        for i in range(1, int(N-2)):
            right_side[i] = np.polyval(np.polyder(poly, 3), x_mesh[i])

        u_hat = np.linalg.solve(operator_3rd.todense(), right_side)

        for i in range(0, int(N)):
            error[i] = np.absolute(u_hat[i]-u_true[i])

        RMS = np.sqrt(np.mean(error**2))
        dx_inverse_list.append(1/dx)
        RMS_list.append(RMS)
        dx = dx*0.5
        N = 0.6/dx+1



    plt.loglog(dx_inverse_list, RMS_list, '-', label='RMS error')
    plt.loglog(dx_inverse_list, [x**-1 for x in dx_inverse_list], '-.', label="1st-order")
    plt.loglog(dx_inverse_list, [x**-2 for x in dx_inverse_list], '-.', label="2nd-order")
    plt.loglog(dx_inverse_list, [x**-3 for x in dx_inverse_list], '-.', label="3rd-order")
    plt.loglog(dx_inverse_list, [x**-4 for x in dx_inverse_list], '-.', label="4th-order")
    plt.loglog(dx_inverse_list, [x**-5 for x in dx_inverse_list], '-.', label="5th-order")
    plt.loglog(dx_inverse_list, [x**-6 for x in dx_inverse_list], '-.', label="6th-order")
    plt.legend(loc='lower left')
    plt.ylabel("RMS error")
    plt.xlabel("dx_inverse")
    plt.title("relationship between RMS error and dx_inverse(3rd poly-fitting scheme)")
    plt.savefig("/Users/Tao/Desktop/ME614/howework/homework1/figures/relationship between RMS error and dx_inverse(3rd poly-fitting scheme.png")
    i=1
    if i==1 :
        break


while Problem22 == 1:

    N = 10
    dx = 0.6 / (N - 1)
    print dx
    x_mesh = np.linspace(-0.3, 0.3, N)

    dx_inverse_list = list()
    RMS_list = list()
    j = 0

    while dx > 0.00005:
        j = j+1
        print j
        x_mesh = np.linspace(-0.3, 0.3, N)

        first_row_5th = np.zeros(N)
        second_row_5th = np.zeros(N)
        last_row_5th = np.zeros(N)

        first_row_5th[0] = 1
        last_row_5th[-6:] = sd.Generate_Weights(x_mesh[-6:], x_mesh[N-1], 1)
        second_row_5th[:6] = sd.Generate_Weights(x_mesh[:6], x_mesh[0], 1)

        operator_5th = gso.Generate_Spatial_Operators(x_mesh, "5th-order", 3)
        operator_5th[0, :] = first_row_5th
        operator_5th[1, :] = second_row_5th
        operator_5th[N-1, :] = last_row_5th

        right_side = np.zeros(N)
        u_true = np.zeros(N)
        error = np.zeros(N)
        poly = np.poly1d([2, 3, 4, 5, 6])  # this is the function I defined

        for i in range(int(N)):
            u_true[i] = np.polyval(poly, x_mesh[i])

        right_side[0] = np.polyval(poly, x_mesh[0])
        right_side[1] = np.polyval(np.polyder(poly, 1), x_mesh[0])
        right_side[N-1] = np.polyval(np.polyder(poly, 1), x_mesh[N-1])

        for i in range(2, int(N-1)):
            right_side[i] = np.polyval(np.polyder(poly, 3), x_mesh[i])

        u_hat = np.linalg.solve(operator_5th.todense(), right_side)

        for i in range(0, int(N)):
            error[i] = np.absolute(u_hat[i]-u_true[i])

        RMS = np.sqrt(np.mean(error**2))
        dx_inverse_list.append(1/dx)
        RMS_list.append(RMS)
        dx = dx*0.7
        N = 0.6/dx+1



    plt.loglog(dx_inverse_list, RMS_list, '-', label='RMS error')
    plt.loglog(dx_inverse_list, [x**-1 for x in dx_inverse_list], '-.', label="1st-order")
    plt.loglog(dx_inverse_list, [x**-2 for x in dx_inverse_list], '-.', label="2nd-order")
    plt.loglog(dx_inverse_list, [x**-3 for x in dx_inverse_list], '-.', label="3rd-order")
    plt.loglog(dx_inverse_list, [x**-4 for x in dx_inverse_list], '-.', label="4th-order")
    plt.loglog(dx_inverse_list, [x**-5 for x in dx_inverse_list], '-.', label="5th-order")
    plt.loglog(dx_inverse_list, [x**-6 for x in dx_inverse_list], '-.', label="6th-order")
    plt.legend(loc='lower left')
    plt.ylabel("RMS error")
    plt.xlabel("dx_inverse")
    plt.title("relationship between RMS error and dx_inverse(5th poly-fitting scheme)")
    plt.savefig("/Users/Tao/Desktop/ME614/howework/homework1/figures/relationship between RMS error and dx_inverse(5th order poly-fitting scheme.png")
    i=1
    if i==1 :
        break