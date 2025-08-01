from firedrake import *
from irksome.ButcherTableaux import GaussLegendre
from irksome.deriv import Dt
from irksome.tools import MeshConstant
from irksome.stepper import TimeStepper
from ufl.algorithms.ad import expand_derivatives
from firedrake.pyplot import tripcolor
import matplotlib.pyplot as plt
import utils
import os

# We will create the Butcher tableau for the lowest-order Gauss-Legendre
# Runge-Kutta method, which is more commonly known as the implicit
# midpoint rule::

butcher_tableau = GaussLegendre(1)
ns = butcher_tableau.num_stages

# Now we define the mesh and piecewise linear approximating space in
# standard Firedrake fashion::

N = 100
x1 = 1.0
y1 = 1.0
T = 2.0

msh = RectangleMesh(N, N, x1, y1)
V = FunctionSpace(msh, "CG", 1)

MC = MeshConstant(msh)
dt = MC.Constant(0.01 / N) #CHANGED
t = MC.Constant(0.0)

x, y = SpatialCoordinate(msh)

u = Function(V)
u.interpolate(cos(pi*x)*cos(pi*y) + 1)

v = TestFunction(V)
F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx

luparams = {"mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu"}


stepper = TimeStepper(F, butcher_tableau, t, dt, u,
                      solver_parameters=luparams)


out_folder = "./sim_outputs/heat_figs_irk/" + utils.dt_now()
os.mkdir(out_folder)

def save_frame(u, t):
    fig, ax = plt.subplots()
    tripcolor(u, axes = ax)
    fig.savefig(f"{out_folder}/heat_{t:.02f}.png")
    plt.close()

i = 0
while (float(t) < T):
    stepper.advance()
    print(float(t))
    t.assign(float(t) + float(dt))
    if i % 100 == 0:
        # It'd be nice if I could give this to another process to start doing in the background while I continue my simulation
        save_frame(u,float(t))
    i += 1

