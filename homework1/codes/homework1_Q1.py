import os
import sys
from numpy import *
import numpy as np   # library to handle arrays like Matlab
import scipy.sparse as scysparse
from pdb import set_trace as keyboard # pdb package allows you to interrupt the python script with the keyboard() command
import spatial_discretization as sd # like include .h file in C++, calls another file
import matplotlib.pyplot as plt

def plot_collocated_centered(N):
    dx = 1  # dx from 1 to almost 0
    global error
    error = list()
    global spacestep
    spacestep_inverse = list()
    global error_percent
    error_percent = list()

    while dx > 0.000000001:
        L = (N) * dx  # from -L to L with N points

        x_stencil = np.linspace(-L/2, L/2, N)  # linespace(start, end, number of elements)

        x_eval = 0.0  # we want to evaluate the deriva  tive of point where x=0
        f = np.tanh(x_stencil) * np.sin(5 * x_stencil + 1.5)  # the function we want to derive : f
        dfdx_analytical = 1 / np.cosh(x_eval) ** 2 * np.sin(5 * x_eval + 1.5) + 5 * np.tanh(x_eval) * np.cos(
            5 * x_eval + 1.5)  # analytical derivative of function f

        derivation_order = 1

        w_der0 = sd.Generate_Weights(x_stencil, x_eval, derivation_order)

        dfdx_hat = w_der0.dot(f)
        discretization_error = np.abs(dfdx_hat - dfdx_analytical)
        discretization_error_percent = discretization_error / dfdx_analytical * 100.

        error.append(discretization_error)
        error_percent.append(discretization_error_percent)
        spacestep_inverse.append(1 / dx)

        dx =dx*0.8
    # reference = [x ** -4 for x in spacestep_inverse]

    plt.loglog(spacestep_inverse, error, linestyle="-", label="Truncation error")

    plt.loglog(spacestep_inverse, [x ** -1 for x in spacestep_inverse], linestyle="-.", label='1st order')
    plt.loglog(spacestep_inverse, [x ** -2 for x in spacestep_inverse], linestyle="-.", label='2nd order')
    plt.loglog(spacestep_inverse, [x ** -3 for x in spacestep_inverse], linestyle="-.", label="3rd order")
    plt.loglog(spacestep_inverse, [x ** -4 for x in spacestep_inverse], linestyle="-.", label="4th order")
    plt.loglog(spacestep_inverse, [x ** -5 for x in spacestep_inverse], linestyle="-.", label="5th order")
    plt.loglog(spacestep_inverse, [x ** -6 for x in spacestep_inverse], linestyle="-.", label='6th order')
    plt.legend(loc='lower left')


    plt.xlabel("The inverse of the grid spacing")
    # plt.legend(loc='lower left', numpoints=1)


    # plt.loglog(spacestep_inverse, [x ** -4 for x in spacestep_inverse], '-.')


def plot_collocated_biased(N):

    dx = 1  # dx from 1 to almost 0
    global x_eval
    x_eval = 0.0  # we want to evaluate the deriva  tive of point where x=0
    global error
    error = list()
    global spacestep
    spacestep_inverse = list()
    global error_percent
    error_percent = list()

    while dx > 0.000000001:
        L = N * dx  # from -L to L with N points

        x_stencil = np.linspace(0, L, N+1)  # linespace(start, end, number of elements)

        f = np.tanh(x_stencil) * np.sin(5 * x_stencil + 1.5)  # the function we want to derive : f
        dfdx_analytical = 1 / np.cosh(x_eval) ** 2 * np.sin(5 * x_eval + 1.5) + 5 * np.tanh(x_eval) * np.cos(
            5 * x_eval + 1.5)  # analytical derivative of function f

        derivation_order = 1

        w_der0 = sd.Generate_Weights(x_stencil, x_eval, derivation_order)

        dfdx_hat = w_der0.dot(f)
        discretization_error = np.abs(dfdx_hat - dfdx_analytical)
        discretization_error_percent = discretization_error / dfdx_analytical * 100.

        error.append(discretization_error)
        error_percent.append(discretization_error_percent)
        spacestep_inverse.append(1 / dx)

        dx =dx*0.8

    plt.loglog(spacestep_inverse, error, linestyle="-", label="Truncation error")
    # plt.loglog(spacestep_inverse, [x ** -1 for x in spacestep_inverse], linestyle="-.", label='1st order')
    plt.loglog(spacestep_inverse, [x ** -1 for x in spacestep_inverse], linestyle="-.", label='1st order')
    plt.loglog(spacestep_inverse, [x ** -2 for x in spacestep_inverse], linestyle="-.", label='2nd order')
    plt.loglog(spacestep_inverse, [x ** -3 for x in spacestep_inverse], linestyle="-.", label="3rd order")
    plt.loglog(spacestep_inverse, [x ** -4 for x in spacestep_inverse], linestyle="-.", label="4th order")
    plt.loglog(spacestep_inverse, [x ** -5 for x in spacestep_inverse], linestyle="-.", label="5th order")
    plt.loglog(spacestep_inverse, [x ** -6 for x in spacestep_inverse], linestyle="-.", label='6th order')
    plt.legend(loc='lower left')

    plt.xlabel("The inverse of the grid spacing")
    # plt.legend(loc='lower left')


def plot_staggered_centered(N):
    dx = 1  # dx from 1 to almost 0
    global error
    error = list()
    global spacestep
    spacestep_inverse = list()
    global error_percent
    error_percent = list()

    while dx > 0.000000001:
        L = (N-1) * dx  # the length of domain

        x_stencil = np.linspace(-L/2, L/2, N)  # linespace(start, end, number of elements)

        x_eval = 0.0  # we want to evaluate the deriva  tive of point where x=0
        f = np.tanh(x_stencil) * np.sin(5 * x_stencil + 1.5)  # the function we want to derive : f
        dfdx_analytical = 1 / np.cosh(x_eval) ** 2 * np.sin(5 * x_eval + 1.5) + 5 * np.tanh(x_eval) * np.cos(
            5 * x_eval + 1.5)  # analytical derivative of function f

        derivation_order = 1

        w_der0 = sd.Generate_Weights(x_stencil, x_eval, derivation_order)

        dfdx_hat = w_der0.dot(f)
        discretization_error = np.abs(dfdx_hat - dfdx_analytical)
        discretization_error_percent = discretization_error / dfdx_analytical * 100.

        error.append(discretization_error)
        error_percent.append(discretization_error_percent)
        spacestep_inverse.append(1 / dx)

        dx =dx*0.8

    plt.loglog(spacestep_inverse, error, linestyle="-", label="Truncation error")
    # plt.loglog(spacestep_inverse, [x ** -1 for x in spacestep_inverse], linestyle="-.", label='1st order')
    plt.loglog(spacestep_inverse, [x ** -1 for x in spacestep_inverse], linestyle="-.", label='1st order')
    plt.loglog(spacestep_inverse, [x ** -2 for x in spacestep_inverse], linestyle="-.", label='2nd order')
    plt.loglog(spacestep_inverse, [x ** -3 for x in spacestep_inverse], linestyle="-.", label="3rd order")
    plt.loglog(spacestep_inverse, [x ** -4 for x in spacestep_inverse], linestyle="-.", label="4th order")
    plt.loglog(spacestep_inverse, [x ** -5 for x in spacestep_inverse], linestyle="-.", label="5th order")
    plt.loglog(spacestep_inverse, [x ** -6 for x in spacestep_inverse], linestyle="-.", label='6th order')
    plt.legend(loc='lower left')

    plt.xlabel("The inverse of the grid spacing")
    # plt.legend(loc='lower left')


def plot_staggered_biased(N):
    dx = 1  # dx from 1 to almost 0
    global x_eval
    x_eval = 0.0  # we want to evaluate the deriva  tive of point where x=0
    global error
    error = list()
    global spacestep
    spacestep_inverse = list()
    global error_percent
    error_percent = list()

    while dx > 0.000000001:
        L = (N-1) * dx  # L is the length of domain

        x_stencil = np.linspace(-dx/2, L-dx/2, N)  # linespace(start, end, number of elements)

        f = np.tanh(x_stencil) * np.sin(5 * x_stencil + 1.5)  # the function we want to derive : f
        dfdx_analytical = 1 / np.cosh(x_eval) ** 2 * np.sin(5 * x_eval + 1.5) + 5 * np.tanh(x_eval) * np.cos(
            5 * x_eval + 1.5)  # analytical derivative of function f

        derivation_order = 1

        w_der0 = sd.Generate_Weights(x_stencil, x_eval, derivation_order)

        dfdx_hat = w_der0.dot(f)
        discretization_error = np.abs(dfdx_hat - dfdx_analytical)
        discretization_error_percent = discretization_error / dfdx_analytical * 100.

        error.append(discretization_error)
        error_percent.append(discretization_error_percent)
        spacestep_inverse.append(1 / dx)

        dx =dx*0.8

    plt.loglog(spacestep_inverse, error, linestyle="-", label="Truncation error")
    plt.loglog(spacestep_inverse, [x ** -1 for x in spacestep_inverse], linestyle="-.", label='1st order')
    plt.loglog(spacestep_inverse, [x ** -2 for x in spacestep_inverse], linestyle="-.", label='2nd order')
    plt.loglog(spacestep_inverse, [x ** -3 for x in spacestep_inverse], linestyle="-.", label="3rd order")
    plt.loglog(spacestep_inverse, [x ** -4 for x in spacestep_inverse], linestyle="-.", label="4th order")
    plt.loglog(spacestep_inverse, [x ** -5 for x in spacestep_inverse], linestyle="-.", label="5th order")
    plt.loglog(spacestep_inverse, [x ** -6 for x in spacestep_inverse], linestyle="-.", label='6th order')
    plt.legend(loc='lower left')
    plt.xlabel("The inverse of the grid spacing")

    # plt.legend(loc='lower left', numpoints=1)
