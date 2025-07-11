import matplotlib.pyplot as plt
from firedrake import quiver, UnitSquareMesh, SpatialCoordinate, VectorFunctionSpace, FunctionSpace, Function, TrialFunction, TestFunction, as_vector, inner, div, dx, DirichletBC, Constant, MixedVectorSpaceBasis, VectorSpaceBasis, solve, errornorm, pi, sin, cos, grad
from firedrake.pyplot import tripcolor
from firedrake.pyplot.mpl import plot
from firedrake.petsc import PETSc
import os
import sys
from pyop2.mpi import COMM_WORLD
import numpy as np
from datetime import datetime
import os

n = 20
dt = 1.0/(n**2)
T = 10.0

mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, 'CG', 1)
u = Function(V)
u_ = Function(V)
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)
ic = Function(V).interpolate(cos(pi*x)*cos(pi*y) + 1)

# We're using a backward Euler scheme. 
# set the initial condition as the starting value for u and the guess for the next u
u_.assign(ic)
u.assign(ic)

f = Function(V).interpolate(Constant(0.0))

a = (inner((u - u_)/dt, v) + inner(grad(u), grad(v)))*dx


out_folder = "heat_figs/" + dt_now
os.mkdir(out_folder)

def save_frame(u, t):
    fig, ax = plt.subplots()
    tripcolor(u, axes = ax)
    fig.savefig(f"{out_folder}/heat_{t:.02f}.png")
    plt.close()

t = 0.0
i=0
while (t <= T):
    solve(a==0, u)
    u_.assign(u)
    t += dt
    if i % 100 == 0:
        save_frame(u,t)

    i+=1
