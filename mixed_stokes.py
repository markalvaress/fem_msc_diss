# example from https://www.wias-berlin.de/people/john/LEHRE/NUMERIK_IV_21_22/num_linear_saddle_prob_3.pdf p13
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import tripcolor
from firedrake.petsc import PETSc
import os
import sys
from pyop2.mpi import COMM_WORLD
from argparse import ArgumentParser, Namespace

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("N_min", help = "Lowest resolution for mesh", type = int)
    parser.add_argument("N_max", help = "Highest resolution for mesh", type = int)
    parser.add_argument("step", help = "Size of jumps to take between N_min and N_max. Must be a divisor of (N_max - N_min).", type = int)
    parser.add_argument("-e", "--elements", help = "Type of elements to use, currently accepts only 'TH' (Taylor-Hood) or 'SV' (Scott-Vogelius).", type = str, nargs = '?', default = "TH")
    parser.add_argument("-k", help = "Polynomial order for largest element", type = int, nargs = '?', default = 2)
    return parser

def validate_args(args: Namespace) -> None:
    if args.elements not in ['TH', 'SV']:
        raise ValueError(f"Element `{args.elements}` not supported: elements argument must be 'TH' (Taylor-Hood) or 'SV' (Scott-Vogelius).")
    if args.k < 2:
        raise ValueError("k must be an integer at least 2.")
    if args.N_max < args.N_min:
        raise ValueError("N_high must be less than N_min.")
    if args.step < 0:
        raise ValueError("step must be a non-negative integer.")
    if (args.step != 0) and ((args.N_max - args.N_min) % args.step != 0):
        raise ValueError("step must be a divisor of (N_high - N_min).")

    return

def plot_and_save(u_: Function, p_: Function, filename: str) -> None:
        fig, ax = plt.subplots(2, figsize = (10,10))
        quiver(u_, axes = ax[0])
        #colors = tripcolor(u, axes = ax[0])
        #fig.colorbar(colors)
        ax[0].set_title("Velocity field")

        colors = tripcolor(p_, axes = ax[1])
        fig.colorbar(colors)
        ax[1].set_title("Pressure field")

        plt.savefig(filename)

def define_and_solve(N: int, elements, k) -> None:
    print(f"{N=}")

    # Define the mesh and function spaces
    mesh = UnitSquareMesh(N, N)
    x,y = SpatialCoordinate(mesh)

    if elements == "TH":
        V = VectorFunctionSpace(mesh, "CG", k)
        Q = FunctionSpace(mesh, "CG", k-1)
    elif elements == "SV":
        V = VectorFunctionSpace(mesh, "CG", k)
        Q = FunctionSpace(mesh, "DG", k-1)
    else:
        raise ValueError(f"Unsupported element {elements}. See documentation.")

    Z = V * Q

    # Define the true solutions
    u_true = Function(V).interpolate(as_vector([
        1000*x**2*(1-x)**4*y**2*(1-y)*(3-5*y),
        -2000*x*(1-x)**3*(1-3*x)*y**3*(1-y)**2
    ]))

    p_true = Function(Q).interpolate(
        pi**2*(x*y**3*cos(2*pi*x**2*y) - x**2*y*sin(2*pi*x*y)) + 1/8
    )

    # Define trial (solution) and test functions
    u,p = TrialFunctions(Z)
    v,q = TestFunctions(Z)

    # Define the RHS of the PDEs. We will add the two equations together to get our governing variational equation.
    lapl_u = as_vector([
        2000*(-1 + x)**2*(y**2*(3 - 8*y + 5*y**2) - 10*x*y**2*(3 - 8*y + 5*y**2) - 6*x**3*(1 - 8*y + 10*y**2) + 3*x**4*(1 - 8*y + 10*y**2) + 3*x**2*(1 - 8*y + 25*y**2 - 40*y**3 + 25*y**4)),
        -4000*(-1 + x)*y*(6*(-1 + y)**2*y**2 - 7*x**3*(3 - 12*y + 10*y**2) + x**4*(9 - 36*y + 30*y**2) + x*(-3 + 12*y - 40*y**2 + 60*y**3 - 30*y**4) + 5*x**2*(3 - 12*y + 16*y**2 - 12*y**3 + 6*y**4))
    ])

    grad_p = as_vector([
        -pi**2*y*(2*pi*x**2*y*cos(2*pi*x*y) - y**2*cos(2*pi*x**2*y) + 2*x*(sin(2*pi*x*y) + 2*pi*x*y**3*sin(2*pi*x**2*y))), 
        -pi**2*x*(2*pi*x**2*y*cos(2*pi*x*y) - 3*y**2*cos(2*pi*x**2*y) + x*(sin(2*pi*x*y) + 2*pi*x*y**3*sin(2*pi*x**2*y)))
    ])

    f = Function(V).interpolate(
        -lapl_u + grad_p
    )

    # Define bilinear forms
    a = (inner(grad(u), grad(v)) - inner(p, div(v)) - inner(q, div(u)))*dx
    L = inner(f, v)*dx

    # the BC on the velocity space (first subspace of Z) is v = (0,0) on all boundaries
    bcs = [DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3, 4))]

    # The object where we'll store the solution (u,p)
    up = Function(Z)

    # Define the nullspace of the pressure space to make solution unique
    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm = COMM_WORLD)])

    # Solve the system
    solve(a == L, up, bcs=bcs, nullspace=nullspace)
    u, p = up.subfunctions

    # Plot the numerical and true results
    plot_and_save(u, p, f"stokes_figs/numerical_{elements}_{N=}.png")
    plot_and_save(u_true, p_true, f"stokes_figs/exact_{elements}_{N=}.png")

    print("Velocity error = ", errornorm(u_true, u))
    print("Pressure error = ", errornorm(p_true, p))

    return

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    validate_args(args)    

    if not os.path.exists("./stokes_figs"):
        os.mkdir("./stokes_figs")

    # Solve problem for a range of mesh sizes
    if args.step == 0:
        define_and_solve(args.N_min, args.elements, args.k)
    else:
        for N in range(args.N_min, args.N_max + 1, args.step):
            define_and_solve(N, args.elements, args.k)
