import matplotlib.pyplot as plt
from firedrake import quiver, UnitSquareMesh, IntervalMesh, SpatialCoordinate, VectorFunctionSpace, FunctionSpace, Function, TrialFunction, TestFunction, as_vector, inner, div, dx, DirichletBC, Constant, MixedVectorSpaceBasis, VectorSpaceBasis, solve, errornorm, pi, sin, cos, grad
from firedrake.pyplot import tripcolor
from firedrake.pyplot.mpl import plot
from firedrake.petsc import PETSc
from firedrake.assemble import assemble
import os
import sys
from pyop2.mpi import COMM_WORLD
import numpy as np
from datetime import datetime
import os
import utils

n = 10
dt = 1.0/(n**4) # use 1/n**2 for 1D problem
T = 0.5
DIM = 2
save_every = 100

if DIM == 1:
    mesh = IntervalMesh(n, 1)
    x = SpatialCoordinate(mesh)
    ic_fn = cos(pi*x[0]) + 1
elif DIM == 2:
    mesh = UnitSquareMesh(n, n)
    x, y = SpatialCoordinate(mesh)
    ic_fn = cos(pi*x)*cos(pi*y) + 1
else:
    raise ValueError("DIM must be 1 or 2")

V = FunctionSpace(mesh, 'CG', 1)
u = Function(V)
u_ = Function(V)
v = TestFunction(V)

ic = Function(V).interpolate(ic_fn)

# We're using a backward Euler scheme. 
# set the initial condition as the starting value for u and the guess for the next u
u_.assign(ic)
u.assign(ic)

f = Function(V).interpolate(Constant(0.0))

a = (inner((u - u_)/dt, v) + inner(grad(u), grad(v)))*dx

dt_now = utils.dt_now()
out_folder = "./sim_outputs/heat_figs/" + dt_now
os.mkdir(out_folder)

def save_frame(u, t):
    fig, ax = plt.subplots()
    if DIM == 1:
        plot(u, axes = ax)
        ax.set_ylim(0,2)
    elif DIM == 2:
        color_plot = tripcolor(u, axes = ax, vmin = 0, vmax = 2)
        fig.colorbar(color_plot, ax=ax)
    fig.savefig(f"{out_folder}/heat_{t:.02f}.png")
    plt.close()

E_form = inner(u, 1.0)*dx
Es = []
t = 0.0
i = 0
while (t <= T):
    solve(a==0, u)
    u_.assign(u)
    t += dt
    if i % save_every == 0:
        save_frame(u,t)

    E = float(assemble(E_form))
    Es.append(E)
    i+=1

with open(f"{out_folder}/energy.txt", "w") as f:
    f.write(str(Es))
