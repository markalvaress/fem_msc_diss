# Simulate Euler equation using Rebholz 2007 scheme 

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
n = 6
dt = 1.0/(n**3)
T = 1
save_every = np.inf

#set up solver
lu = {
    "mat_type":"aij",
    "snes_type":"newtonls",
    "ksp_type":"preonly",
    "pc_type":"lu",
    "pc_factor_mat_solver_type":"mumps",
}
sp = lu

# Define mesh
mesh = PeriodicUnitCubeMesh(n,n,n)
x, y, z = SpatialCoordinate(mesh)

# Define function space
X = VectorFunctionSpace(mesh, "CG", 2)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
Z = X*V*Q

# For defining the form. u, w and p are not trial functions because it's nonlinear
uwp = Function(Z) # will hold u, w and p in next time step
uwp_ = Function(Z) # ... in current time step
v,chi,q = TestFunctions(Z)

# Define initial condition satisfying boundary conditions
ic = Function(X).interpolate(as_vector([
    cos(2*pi*z),
    sin(2*pi*z),
    sin(2*pi*x)
]))

# set the initial condition as the starting value for u
uwp_.sub(0).assign(ic)

# Define the nonlinear functional F.
# First define f: this could represent an external force - we set it to 0.
f = Function(V).interpolate(as_vector([Constant(0.0), Constant(0.0), Constant(0.0)]))

u, w, p = split(uwp)
u_, w_, p_ = split(uwp_)

u_half = (u + u_)/2
w_half = (w + w_)/2
p_half = (p + p_)/2
F = (
    inner((u - u_)/dt, v) 
    + inner(cross(w_half, u_half), v)
    - inner(p_half, div(v))
    - inner(q, div(u))
    + inner(w - curl(u), chi)
    # Extra terms from Mingdong discretisation
    #+ inner(grad(p), chi) - inner(div(w), q) 
    - inner(f, v)
)*dx

# Prep output folder
dt_now = utils.dt_now()
out_folder = utils.init_outfolder("euler_rebholz_figs/" + dt_now)

def save_frame(u, t):
    fig, ax = plt.subplots()
    quiver(u, axes = ax)
    fig.savefig(f"{out_folder}/vel_{t:.02f}.png", dpi=500)
    plt.close()

def save_pressure_frame(p,t):
    fig, ax = plt.subplots()
    tripcolor(p, axes = ax)
    fig.savefig(f"{out_folder}/pres_{t:.02f}.png", dpi=500)
    plt.close()

# Define the energy and helicity
E_form = 0.5*inner(u, u)*dx
Es = []
H_form = 0.5*inner(u, curl(u))*dx
Hs = []

# Run simulation with progress bar
t = 0.0
i = 0
bcs = None

pb = NonlinearVariationalProblem(F, uwp, bcs)
solver = NonlinearVariationalSolver(pb, solver_parameters = sp)

with tqdm(total = T) as pbar:
    while (t <= T):
        solver.solve()
        t += dt
        if (i+1) % save_every == 0:
            u, w, p = uwp.subfunctions
            save_frame(u,t)
            save_pressure_frame(p,t)

        E = float(assemble(E_form))
        Es.append(E)
        H = float(assemble(H_form))
        Hs.append(H)

        print("E: ", E)
        print("H: ", H)

        i+=1
        pbar.update(dt)

        uwp_.assign(uwp)

# Save energy history to file
with open(f"{out_folder}/energy.txt", "w") as f:
    f.write(str(Es))

with open(f"{out_folder}/helicity.txt", "w") as f:
    f.write(str(Hs))

# Plot and save energy over time
t_list = [i*dt for i in range(len(Es))]
plt.plot(t_list, Es)
plt.xlabel("$t$")
plt.ylabel(r"$E_h(t)$")
plt.ylim([0, max(Es)+0.5])
plt.savefig(f"{out_folder}/energy.png", dpi=500)

plt.plot(t_list, Hs)
plt.xlabel("$t$")
plt.ylabel(r"$H_h(t)$")
plt.ylim([0, max(Hs)+0.5])
plt.savefig(f"{out_folder}/helicity.png", dpi=500)

utils.done(out_folder)
