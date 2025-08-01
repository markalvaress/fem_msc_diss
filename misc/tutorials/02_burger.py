# this one is gonna be time dependent ooh
from firedrake import *
from firedrake.pyplot import tripcolor, tricontour
import matplotlib.pyplot as plt

n = 30
mesh = UnitSquareMesh(n,n)
V = VectorFunctionSpace(mesh, "CG", 2) # degree 2 continuous lagrange polys
V_out = VectorFunctionSpace(mesh, "CG", 1) # not sure why we need this

# no trial functions because it's a nonlinear problem...
u_ = Function(V, name = "Velocity")
u = Function(V, name = "VelocityNext")

v = TestFunction(V)

# set initial condition
x, y = SpatialCoordinate(mesh) # I CHANGD THIS
ic = project(as_vector([sin(pi*x), 0]), V)

# set the initial condition as the starting value for u and the guess for the next u
u_.assign(ic)
u.assign(ic)

nu = 0.0001 # low viscosity

# timestep produces advective Courant number ~ 1. This is stronger than we need for stability
timestep = 1.0/n

# This is backward Euler
F = (inner((u - u_)/timestep, v) + inner(dot(u, nabla_grad(u)), v) + nu*inner(grad(u), grad(v)))*dx

# prep the output file
outfile = VTKFile("burgers.pvd")

def save_frame(u, t):
    fig, ax = plt.subplots()
    quiver(u, axes = ax)
    fig.savefig(f"figs/burger_quiv_{t:.02f}.png")

# loop over the time steps and save the output
t = 0.0
end = 0.6
while (t <= end):
    solve(F == 0, u)
    u_.assign(u)
    t += timestep
    outfile.write(project(u, V_out, name = "Velocity"))
    save_frame(u,t)

print("Done!")