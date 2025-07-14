from firedrake import *
from firedrake.pyplot import tripcolor, tricontour
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d


# Define mesh and mesh size
N = 10
mesh = UnitSquareMesh(N,N)
x,y = SpatialCoordinate(mesh)
h = np.sqrt(2)*(1/N)

# use piecewise linear functions cts between elements
V = FunctionSpace(mesh, "CG", 1)

# interpolate true solution into function space
u_true = Function(V).interpolate(
    sin(pi*x)*sin(pi*y)
)

# Initialise trial and test functions - used to define the forms below
u = TrialFunction(V)
v = TestFunction(V)

# Set up forms in problem
f = Function(V)
f.interpolate(2*pi**2*sin(pi*x)*sin(pi*y))
a = (inner(grad(u), grad(v))) * dx
L = inner(f,v) * dx
bc = DirichletBC(V, Constant(0), sub_domain = "on_boundary")

# redefine u to be a function holding the sol'n, and compute sol
u = Function(V)
solve(a == L, u, bcs = bc)

# Plot les results
fig, ax = plt.subplots()
colors = tripcolor(u, axes = ax)
fig.colorbar(colors)
fig.savefig("poisson_analyt_hmp.png")
fig.clf()

fig = plt.figure()
axes = fig.add_subplot(projection='3d')
trisurf(u, axes = axes)
fig.savefig("poisson_analyt_surf.png")

# Compute error
u_error = errornorm(u_true, u, norm_type = "H1")
print(f"{u_error=:.2e}")