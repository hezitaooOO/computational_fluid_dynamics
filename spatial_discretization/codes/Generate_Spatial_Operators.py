import numpy as np
import scipy.sparse as scysparse
import spatial_discretization as sd
import sys
from pdb import set_trace as keyboard


def Generate_Spatial_Operators(x_mesh, order_scheme,  der_order):
    N = x_mesh.size
    # you should pre-allocate a sparse matrix predicting already the number of non-zero entries
    A = scysparse.lil_matrix((N, N), dtype=np.float64)  # here N is 10

    if order_scheme == "3rd-order":
        order_poly = 3+1
        for i, x_eval in enumerate(x_mesh):

            if i == 0:
                x_stencil = x_mesh[0:order_poly]  # this includes points 0,1,2
                A[i, i:order_poly] = sd.Generate_Weights(x_stencil, x_mesh[i], der_order)

            # elif i == 1:
            #     x_stencil = x_mesh[:order_poly]
            #     A[i, :order_poly] = sd.Generate_Weights(x_stencil, x_mesh[i], der_order)
            #
            elif i == N - 2:
                x_stencil = x_mesh[-order_poly:]
                A[i, -order_poly:] = sd.Generate_Weights(x_stencil, x_mesh[i], der_order)

            elif i == N - 1:
                x_stencil = x_mesh[-order_poly:]  # this includes points 0,1,2
                A[i, -order_poly:]= sd.Generate_Weights(x_stencil, x_mesh[i], der_order)


            else:
                x_stencil = x_mesh[i-(order_poly/2-1):i+(order_poly/2+1)]
                A[i, i-(order_poly/2-1):i+(order_poly/2+1)] = sd.Generate_Weights(x_stencil, x_mesh[i], der_order)

    if order_scheme == "5th-order":
        order_poly = 5+1
        for i, x_eval in enumerate(x_mesh):

            if i == 0:
                x_stencil = x_mesh[0:order_poly]  # this includes points 0,1,2
                A[i, 0:order_poly] = sd.Generate_Weights(x_stencil, x_stencil[i], der_order)

            elif i == 1:
                x_stencil = x_mesh[0:order_poly]  # this includes points 0,1,2
                A[i, 0:order_poly] = sd.Generate_Weights(x_stencil, x_stencil[i], der_order)

            elif i == N - 1:
                x_stencil = x_mesh[-order_poly:]  # this includes points 0,1,2
                A[i, -order_poly:]= sd.Generate_Weights(x_stencil, x_stencil[order_poly-(N-i)], der_order)

            elif i == N - 2:
                x_stencil = x_mesh[-order_poly:]
                A[i, -order_poly:] = sd.Generate_Weights(x_stencil, x_stencil[order_poly-(N-i)], der_order)

            elif i == N - 3:
                x_stencil = x_mesh[-order_poly:]
                A[i, -order_poly:] = sd.Generate_Weights(x_stencil, x_stencil[order_poly - (N-i)], der_order)

            else:
                x_stencil = x_mesh[i-(order_poly/2-1):i+(order_poly/2+1)]  # this includes points 0,1,2
                A[i, i-(order_poly/2-1):i+(order_poly/2+1)] = sd.Generate_Weights(x_stencil, x_stencil[order_poly/2-1], der_order)

    # print "Here is where you loop over the points of the stencil and deploy your weights in the matrix"

    # convering to csr format, which appears to be more efficient for matrix operations
    return A