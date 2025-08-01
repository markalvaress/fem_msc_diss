# example from https://www.wias-berlin.de/people/john/LEHRE/NUMERIK_IV_21_22/num_linear_saddle_prob_3.pdf p13
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import tripcolor
from firedrake.petsc import PETSc
import os
import sys
from pyop2.mpi import COMM_WORLD
import numpy as np
from datetime import datetime
from scipy.stats import linregress
from utils import dt_now

def init_parser(outfolder_default: str, k_default: int) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("N_min", help = "Lowest resolution for mesh", type = int)
    parser.add_argument("N_max", help = "Highest resolution for mesh", type = int)
    parser.add_argument("step", help = "Size of jumps to take between N_min and N_max. Must be a divisor of (N_max - N_min).", type = int)
    parser.add_argument("-e", "--elements", help = "Type of elements to use, currently accepts only 'TH' (Taylor-Hood), 'div' (divergent element, using CGK^n x CGK) or 'SV' (Scott-Vogelius).", type = str, nargs = '?', default = "TH")
    parser.add_argument("-k", help = "Polynomial order(s) for largest element", type = int, nargs = '*', default = [k_default])
    parser.add_argument("-o", "--outputfolder", help = "Directory to save output to", type = str, nargs = '?', default = outfolder_default)
    parser.add_argument("-f", "--figs", help = "Store all figures, defaults to false.", action = "store_true")
    return parser

def validate_args(args: Namespace) -> None:
    if args.elements not in ['TH', 'SV', 'div']:
        raise ValueError(f"Element `{args.elements}` not supported: elements argument must be 'TH' (Taylor-Hood), 'div' (divergent), or 'SV' (Scott-Vogelius).")
    if (min(args.k) < 1) and (args.elements != 'div'):
        raise ValueError("k must consist of integers greater than or equal to 1.")
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

def define_and_solve(N: int, elements: str, k: int, output_folder: str, store_figs: bool) -> list[float, float, float]:
    """Return the max triangle diameter, velocity error, and pressure error."""
    print(f"{N=}, {k=}")

    # Calculate max triangle diameter, which will be the hypotenuse of the right triangle with side lengths 1/N. 
    h = np.sqrt(2)*(1/N)

    # Define the mesh and function spaces
    mesh = UnitSquareMesh(N, N)
    x,y = SpatialCoordinate(mesh)

    if elements == "TH":
        V = VectorFunctionSpace(mesh, "CG", k)
        Q = FunctionSpace(mesh, "CG", k-1)
    elif elements == "SV":
        V = VectorFunctionSpace(mesh, "CG", k)
        Q = FunctionSpace(mesh, "DG", k-1)
    elif elements == "div":
        V = VectorFunctionSpace(mesh, "CG", k)
        Q = FunctionSpace(mesh, "CG", k)
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
    if store_figs:
        plot_and_save(u, p, f"{output_folder}/numerical_{elements}_{N=}.png")
        plot_and_save(u_true, p_true, f"{output_folder}/exact_{elements}_{N=}.png")

    u_error = errornorm(u_true, u, norm_type = "H1")
    p_error = errornorm(p_true, p, norm_type = "L2")

    print(f"{h=:.2e}")
    print(f"Velocity error = {u_error:.2e}")
    print(f"Pressure error = {p_error:.2e}")

    return [float(h), float(u_error), float(p_error)]

def create_err_fig(h_ks: list | np.ndarray, errs: list | np.ndarray, out_folder: str, quantity: str, quantity_short: str, norm: str) -> float:
    """Returns gradient of log-log plot."""
    hs, ks = zip(*h_ks)
        
    # calc gradient
    lr_results = linregress(np.log(hs), np.log(errs))
    grad = lr_results.slope

    plt.clf()
    plt.loglog(hs, errs)
    plt.scatter(hs, errs)
    plt.xlabel(r"$\log h$")
    plt.ylabel(rf"$\log \|{quantity_short}-{quantity_short}_h\|_" + "{" + norm  + "}$")
    plt.title(f"{quantity} convergence, slope = {grad:.2f}")
    plt.savefig(f"{out_folder}/{quantity_short}_error.png")
    return float(grad)

def main(args):
    time_now = dt_now()
    out_folder = args.outputfolder + "/" + time_now

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # prepare to store the error results
    h_ks = []
    u_errs = []
    p_errs = []

    # just so I can still use range(start, stop, step) below
    if args.step == 0:
        step = 1
    else:
        step = args.step

    # Solve problem for a range of mesh sizes and polynomial orders
    for N in range(args.N_min, args.N_max + 1, step):
        for k in args.k:
            h, u_err, p_err = define_and_solve(N, args.elements, k, out_folder, args.figs)
            h_ks.append((h,k))
            u_errs.append(u_err)
            p_errs.append(p_err)

    if (args.step > 0) and (len(args.k) == 1):
        # TODO: make better plotting function if I have multiple h and multiple k
        grad_u = create_err_fig(h_ks, u_errs, out_folder, "Velocity", "u", r"H^1(\Omega)")  
        grad_p = create_err_fig(h_ks, p_errs, out_folder, "Pressure", "p", r"L^2(\Omega)") 
    else:
        grad_u, grad_p = None, None 

    with open(f"{out_folder}/sim_output.txt", "w") as f:
        f.writelines([
            "Date & time: " + time_now + "\n",
            f"Params: {args.__str__().replace("Namespace", "")}" + "\n",
            "h_ks = " + str(h_ks) + "# pairs of (h,k) \n",
            "u_errs = " + str(u_errs) + "\n",
            "p_errs = " + str(p_errs) + "\n",
            "velocity convergence rate = " + str(grad_u) + "\n",
            "pressure convergence rate = " + str(grad_p) + "\n"
        ])

    print("Done")

if __name__ == "__main__":
    parser = init_parser(outfolder_default="./sim_outputs/stokes_sims", k_default = 2)
    args = parser.parse_args()
    validate_args(args)

    main(args)
        
