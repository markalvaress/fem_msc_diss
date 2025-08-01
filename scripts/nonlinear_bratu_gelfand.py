# Solve the Bratu-Gelfand equation u'' + lmbda e^u = 0, subject to u(0) = 0 = u(1).
# Using Newton-Kantorovich algorithm.

from firedrake import *
from firedrake.pyplot.mpl import plot
import matplotlib.pyplot as plt

lmbda = 2.0
n_iters = 10

mesh = UnitIntervalMesh(200)
x = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)

du = TrialFunction(V)
v = TestFunction(V)
bc = DirichletBC(V, Constant(0), sub_domain = "on_boundary")

# our initial guess
u = Function(V).interpolate(Constant(0))

# where we will store the solution du
du_update = Function(V)

def save_plot(u, i):
    fig, ax = plt.subplots()
    plot(u, axes = ax)
    fig.savefig(f"./sim_outputs/bratu_figs/bratu_{i}.png")
    plt.close()

# Iterates over linear approximations of problem to find solution. Save a plot of each u_n.
for i in range(n_iters):
    save_plot(u,i)
    a = -inner(grad(du), grad(v))*dx + inner(lmbda*exp(u)*du, v)*dx
    F = inner(grad(u), grad(v))*dx - inner(lmbda*exp(u), v)*dx

    solve(a == F, du_update, bcs = bc)
    u = Function(V).interpolate(u + du_update)


