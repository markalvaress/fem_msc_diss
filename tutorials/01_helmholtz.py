# https://www.firedrakeproject.org/demos/helmholtz.py.html
# Solving: -∇^2 u + u = f, dot(grad(u), n) = 0 on Gamma, Omega is unit square.
from firedrake import *
from firedrake.pyplot import tripcolor, tricontour
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(10,10)

# use piecewise linear functions cts between elements
V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

# define rhs function
f = Function(V)
x,y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
a = (inner(grad(u), grad(v)) + inner(u,v)) * dx
L = inner(f,v) * dx

# REDEFINE u to be a function holding the sol'n (why?)
u = Function(V)
# compute solution
solve(a == L, u, solver_parameters = {'ksp_type =': 'cg', 'pc_type': 'none'})
fig, ax = plt.subplots()
colors = tripcolor(u, axes = ax)
fig.colorbar(colors)
fig.savefig("bongus.png")

fig, axes = plt.subplots()
contours = tricontour(u, axes=axes)
fig.colorbar(contours)
fig.savefig("bongus_contours.png")