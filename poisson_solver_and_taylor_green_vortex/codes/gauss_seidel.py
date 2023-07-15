import numpy as np
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg
from pdb import set_trace as keyboard




def gs_sor(A, b, w=1.4, tol=1.0e-7):

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
    # inv = spysparselinalg.inv(D + w*L)


    while r / r0 > tol:
        # print r / r0
        index += 1
        # print index
        A1 = inv.solve(((1 - w) * D - w * U).dot(phi_old))
        A2 = w*inv.solve(b)
        phi = A1 + A2

        r = np.linalg.norm(A.dot(phi) - b)
        phi_old = phi

    return phi, index



def gs_sor_nonlinear(Derivative, DivGrad, b, w = 1.0):

    tol = 1.0e-10
    n = Derivative.shape[0]
    A = DivGrad
    phi_0 = np.ones(n)
    phi = phi_0
    index = 0
    RHS = phi * (Derivative.dot(phi)) - b

    norm_0 = np.linalg.norm(A.dot(phi) - RHS)
    print norm_0

    while True:

        print index
        index += 1
        # phi, it = gs_sor(A, b, w)
        phi_old = phi
        phi = spysparselinalg.spsolve(A, RHS)
        RHS = phi * (Derivative.dot(phi)) - b
        # norm_k = np.linalg.norm(A.dot(phi) - b)
        norm_k = np.linalg.norm(phi-phi_old)

        print norm_k/norm_0
        # if (norm_k) < tol:
        #    return phi, index
        if (norm_k / norm_0) < tol:
            return phi, index



# A = np.array([[3.9, -2.1, 1.3], [1.9, -3.1, 2.3], [-1.0, 2.0, 6.0]])
# b = np.array([1.0, 2.0, 3.0])
# n = 20
#
# print A.shape
# print b.shape
#
#
# print("\n\ninit"),
# print("")
# x = jacobi_SOR(A, b, omega=1)
# print("\nSol "),
# print(x)
# print("Act "),
# print solve(A, b)
# print("\n")