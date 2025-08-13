# Solve the Bratu-Gelfand equation u'' + lmbda e^u = 0, subject to u(0) = 0 = u(1).
# Using Newton-Kantorovich algorithm.

from firedrake import *
from firedrake.pyplot.mpl import plot
import matplotlib.pyplot as plt
import utils
import scienceplots
import matplotlib
from argparse import ArgumentParser
from num2words import num2words
matplotlib.use('Agg')
plt.style.use("science")

parser = ArgumentParser()
parser.add_argument("converge_to", help = "Converge to `lower` or `upper` solution.", type = str)
parser.add_argument("n_iters", help = "Number of iterations", type = int)
args = parser.parse_args()

time_now = utils.dt_now()
out_folder = utils.init_outfolder("bratu_figs/" + time_now)

lmbda = 2.0

mesh = UnitIntervalMesh(200)
x = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)

du = TrialFunction(V)
v = TestFunction(V)
bc = DirichletBC(V, Constant(0), sub_domain = "on_boundary")

# our initial guess
if args.converge_to == "lower":
    u = Function(V).interpolate(Constant(0))
    alpha = 0.5894
elif args.converge_to == "upper":
    u = Function(V).interpolate(4*sin(pi*x[0]))
    alpha = 2.1268
else:
    raise ValueError("converge_to must be 'lower' or 'upper'.")

u_true = Function(V).interpolate(2*ln(cosh(alpha)/cosh(alpha*(1-2*x[0]))))

# where we will store the solution du
du_update = Function(V)

# Iterates over linear approximations of problem to find solution. Save a plot of each u_n.
fig, ax = plt.subplots(figsize = (8,6))

for i in range(args.n_iters):
    plot(u, axes = ax, label = f"{num2words(i, to = "ordinal_num")} approximation")
    a = -inner(grad(du), grad(v))*dx + inner(lmbda*exp(u)*du, v)*dx
    F = inner(grad(u), grad(v))*dx - inner(lmbda*exp(u), v)*dx

    solve(a == F, du_update, bcs = bc)
    u = Function(V).interpolate(u + du_update)

plot(u_true, axes = ax, label = "True solution")
ax.legend()
# get some axis labels up in this BLEEP
fig.savefig(f"{out_folder}/bratu.png", dpi=300)
utils.done(out_folder)