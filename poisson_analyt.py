from mixed_stokes import init_parser, validate_args, create_err_fig
from firedrake import *
from firedrake.pyplot import tripcolor, tricontour, trisurf
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from utils import dt_now
import os

def plot_and_save(u: Function, filename: str, what: str):
    assert what in ["colours", "surface", "both"]

    if what in ["colours", "both"]:
        fig, ax = plt.subplots()
        colors = tripcolor(u, axes = ax)
        fig.colorbar(colors)
        fig.savefig(f"{filename}_hmp.png")
        fig.clf()

    if what in ["surface", "both"]:
        fig = plt.figure()
        axes = fig.add_subplot(projection='3d')
        trisurf(u, axes = axes)
        fig.savefig(f"{filename}_surf.png")
    
    return

def define_and_solve(N: int, k: int, output_folder: str, store_figs: bool, norm_type: str) -> list[float, float]:
    mesh = UnitSquareMesh(N,N)
    x,y = SpatialCoordinate(mesh)
    h = np.sqrt(2)*(1/N)

    # use cts functions of piecewise degree k
    V = FunctionSpace(mesh, "CG", k)

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

    if store_figs:
        plot_and_save(u, output_folder + f"/poisson_{N}_{k}", "both")

    # Compute error
    u_error = errornorm(u_true, u, norm_type = norm_type)
    return [float(h), float(u_error)]

def latexify_errornorm(error_norm):
    return error_norm[0] + "^" + error_norm[1] + r"(\Omega)"

def main(args):
    time_now = dt_now()
    out_folder = args.outputfolder + "/" + time_now

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # prepare to store the error results
    h_ks = []
    u_errs = []

    # just so I can still use range(start, stop, step) below
    if args.step == 0:
        step = 1
    else:
        step = args.step

    for N in range(args.N_min, args.N_max + 1, step):
        for k in args.k:
            h, u_err = define_and_solve(N, k, out_folder, args.figs, args.error_norm)
            h_ks.append((h,k))
            u_errs.append(u_err)

    if (args.step > 0) and (len(args.k) == 1):
        # TODO: make better plotting function if I have multiple h and multiple k
        grad_u = create_err_fig(h_ks, u_errs, out_folder, "u", "u", latexify_errornorm(args.error_norm))
    else:
        grad_u = None 

    with open(f"{out_folder}/sim_output.txt", "w") as f:
        f.writelines([
            "Date & time: " + time_now + "\n",
            f"Params: {args.__str__().replace("Namespace", "")}" + "\n",
            "h_ks = " + str(h_ks) + "# pairs of (h,k) \n",
            "u_errs = " + str(u_errs) + "\n",
            "velocity convergence rate = " + str(grad_u) + "\n"
        ])

    print("Done")

    

if __name__ == "__main__":
    parser = init_parser(outfolder_default = "./poisson_analyt_sims", k_default = 1)
    parser.add_argument("-error_norm", help = "Error norm to use, either 'H1' or 'L2'.", type = str, default = "H1")
    args = parser.parse_args()
    validate_args(args)

    main(args)