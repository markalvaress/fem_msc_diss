# Simulate Euler equation using mixed formulation and a backward Euler timestepping scheme

import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import tripcolor, quiver
from firedrake.pyplot.mpl import plot
from firedrake.petsc import PETSc
from firedrake.assemble import assemble
from pyop2.mpi import COMM_WORLD
import numpy as np
from datetime import datetime
from tqdm import tqdm
import utils
import scienceplots
import matplotlib
matplotlib.use('Agg')
plt.style.use("science")

# Simulation parameters
n = 10
dt = 1.0/(n**4)
T = 0.5
save_every = np.inf

# Define mesh
mesh = UnitSquareMesh(n, n)
x, y = SpatialCoordinate(mesh)

# Define function space: Taylor-hood [CG2]^n x [CG1] elements
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
Z = V*Q

# For defining the form. u and p are not trial functions because it's nonlinear
up = Function(Z) # will hold u and p in next time step
up_ = Function(Z) # ... in current time step
v,q = TestFunctions(Z)

# Define initial condition satisfying boundary conditions
ic = Function(V).interpolate(as_vector([
    -2*pi*sin(pi * x)**2 *sin(pi*y) * cos(pi * y),
    2*pi*sin(pi * y)**2 * sin(pi * x) * cos(pi * x)
]))

# set the initial condition as the starting value for u
up_.sub(0).assign(ic)

# Define the nonlinear functional F.
# First define f: this could represent an external force - we set it to 0.
f = Function(V).interpolate(as_vector([Constant(0.0), Constant(0.0)]))

u, p = split(up)
u_, p_ = split(up_)
F = (
    inner((u - u_)/dt, v) 
    + inner(dot(u, nabla_grad(u)), v)
    - inner(p, div(v))
    - inner(q, div(u))
    - inner(f, v)
)*dx

# Prep output folder
dt_now = utils.dt_now()
out_folder = utils.init_outfolder("euler_figs/" + dt_now)

def save_frame(u, t):
    fig, ax = plt.subplots()
    quiver(u, axes = ax)
    fig.savefig(f"{out_folder}/vel_{t:.02f}.png", dpi=500)
    plt.close()

# Assert that velocity functions are zero on the boundary
bcs = [DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3, 4))]
# Define the nullspace of the pressure space to make solution unique
nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm = COMM_WORLD)])

# Define the energy
E_form = 0.5*inner(u, u)*dx
Es = []

# Run simulation with progress bar
t = 0.0
i = 0
with tqdm(total = T) as pbar:
    while (t <= T):
        solve(F == 0, up, bcs = bcs, nullspace = nullspace)
        up_.assign(up)
        t += dt
        if i % save_every == 0:
            u, _ = up.subfunctions
            save_frame(u,t)

        E = float(assemble(E_form))
        Es.append(E)
        i+=1
        pbar.update(dt)

# Save energy history to file
with open(f"{out_folder}/energy.txt", "w") as f:
    f.write(str(Es))

# Plot and save energy over time
t_list = [i*dt for i in range(len(Es))]
plt.plot(t_list, Es)
plt.xlabel("$t$")
plt.ylabel(r"$E_h(t)$")
plt.savefig(f"{out_folder}/energy.png", dpi=500)

utils.done(out_folder)