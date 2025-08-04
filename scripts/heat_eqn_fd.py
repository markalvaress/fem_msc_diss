import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import tripcolor
from firedrake.pyplot.mpl import plot
from firedrake.assemble import assemble
import numpy as np
import utils
from argparse import ArgumentParser
import scienceplots
import matplotlib
matplotlib.use('Agg')
plt.style.use("science")

parser = ArgumentParser()
parser.add_argument("-d", "--dimension", help = "Dimension of problem. Must be 1 or 2.", type = int)
parser.add_argument("-f", "--savefigs", help = "Flag to save figures of the heat graph.", action = "store_true")
parser.add_argument("--save-every", type=int, help = "Save figures of the heat graph every ... timestep. Only needed if using --savefigs.", nargs = '?', default = -1)

args = parser.parse_args()
DIM = args.dimension
assert DIM in [1,2]
if args.savefigs:
    if args.save_every == -1:
        raise ValueError("You must supply a positive value for --save-every if you are using --savefigs.")
    else:
        save_every = args.save_every
else:
    save_every = np.inf

n = 10
dt = 1.0/(n**(2*DIM))
T = 0.5

if DIM == 1:
    mesh = IntervalMesh(n, 1)
    x = SpatialCoordinate(mesh)
    ic_fn = cos(pi*x[0]) + 1
elif DIM == 2:
    mesh = UnitSquareMesh(n, n)
    x, y = SpatialCoordinate(mesh)
    ic_fn = cos(pi*x)*cos(pi*y) + 1
else:
    # belt and braces.
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
out_folder = utils.init_outfolder("heat_figs/" + dt_now)

def save_frame(u, t):
    fig, ax = plt.subplots()
    if DIM == 1:
        plot(u, axes = ax)
        ax.set_ylim(0,2)
    elif DIM == 2:
        color_plot = tripcolor(u, axes = ax, vmin = 0, vmax = 2)
        fig.colorbar(color_plot, ax=ax)
    fig.savefig(f"{out_folder}/heat_{t:.02f}.png", dpi = 300)
    plt.close()

E_form = inner(u, 1.0)*dx
Es = []
t = 0.0
i = 0

# Simulation loop
print("Starting simulation")
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

# Create energy-time plot
# TODO: change this so the x-axis has time rather than n 
ns = list(range(len(Es)))
plt.plot(ns, Es)
plt.ylim(0,2)
plt.xlabel("$n$")
plt.ylabel(r"$E(n\Delta t)$")
plt.savefig(f"{out_folder}/energy_vs_time.png", dpi=300)

utils.done(out_folder)