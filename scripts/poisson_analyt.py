from mixed_stokes import init_parser, validate_args, create_err_fig
from firedrake import *
from firedrake.pyplot import tripcolor, tricontour, trisurf
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from utils import dt_now, done, init_outfolder, sim_outputs_folder
from pyop2.mpi import COMM_WORLD
from netgen.geom2d import SplineGeometry
import scienceplots
import matplotlib
from argparse import BooleanOptionalAction
matplotlib.use('Agg')
plt.style.use("science")

def plot_and_save(u: Function, filename: str, what: str):
    """Plot u. Plot either the magnitude of u (colours), u as a surface, or both."""
    assert what in ["colours", "surface", "both"]

    if what in ["colours", "both"]:
        fig, ax = plt.subplots()
        colors = tripcolor(u, axes = ax)
        fig.colorbar(colors)
        fig.savefig(f"{filename}_hmp.png", dpi=500)
        fig.clf()

    if what in ["surface", "both"]:
        fig = plt.figure()
        axes = fig.add_subplot(projection='3d')
        trisurf(u, axes = axes)
        fig.savefig(f"{filename}_surf.png", dpi=500)
    
    return
    
def create_spline_mesh(h: float) -> Mesh:
    # Note: in this case subsequent meshes are not likely to be refinements of earlier meshes, so you might
    # get little jumps where a smaller h gives a worse result.
    geo = SplineGeometry()

    geo.AddRectangle(
        p1 = (0, 0),
        p2 = (1,1),
        bc = "rectangle",
    )

    ngmsh = geo.GenerateMesh(maxh=h)
    msh = Mesh(ngmsh) # make a firedrake mesh
    return msh

def define_and_solve(N: int, k: int, output_folder: str, store_figs: bool, norm_type: str, bc_type: str, symm_soln: bool, regmesh: bool) -> list[float, float]:
    """Solve the poisson problem with the given parameters."""
    h = np.sqrt(2)*(1/N)

    if regmesh:  
        mesh = UnitSquareMesh(N,N)
    else:
        mesh = create_spline_mesh(h)

    x,y = SpatialCoordinate(mesh)
    
    # use continuous functions of piecewise degree k
    V = FunctionSpace(mesh, "CG", k)

    # Initialise trial and test functions - used to define the forms below
    u = TrialFunction(V)
    v = TestFunction(V)

    # Set up forms in problem
    f = Function(V)
    if bc_type == 'dirichlet':
        if symm_soln:
            # interpolate true solution into function space
            u_true = Function(V).interpolate(
                sin(pi*x)*sin(pi*y)
            )
            f.interpolate(2*pi**2*sin(pi*x)*sin(pi*y))
        else:
            u_true = Function(V).interpolate(
                -(x**3 - 5*x**2 + 4*x)*(y**3 + 2*y**2 - 3*y)
            )
            f.interpolate((6*x - 10)*(y**3 + 2*y**2 - 3*y) + (x**3 - 5*x**2 + 4*x)*(6*y + 4))

        bc = DirichletBC(V, Constant(0), sub_domain = "on_boundary")
        nullspace = None
    elif bc_type == 'mixed':
        u_true = Function(V).interpolate(
            cos(pi*x)*sin(pi*y)
        )
        f.interpolate(2*pi**2*cos(pi*x)*sin(pi*y))
        # subdomain is lines y == 0 and y == 1. 
        if regmesh:
            sub_domain = [3,4]
        else:
            sub_domain = [1,3]
        
        bc = DirichletBC(V, Constant(0), sub_domain = sub_domain)
        nullspace = None
    elif bc_type == 'neumann':
        u_true = Function(V).interpolate(
            cos(pi*x)*cos(pi*y)
        )
        f.interpolate(2*pi**2*cos(pi*x)*cos(pi*y))
        bc = None
        nullspace = VectorSpaceBasis(constant = True, comm = COMM_WORLD)
    else:
        raise ValueError('You have made a very grave mistake.')

    a = (inner(grad(u), grad(v))) * dx
    L = inner(f,v) * dx

    # redefine u to be a function holding the solutinn, and compute solution
    u = Function(V)
    solve(a == L, u, bcs = bc, nullspace = nullspace)

    if store_figs:
        plot_and_save(u, output_folder + f"/poisson_{N}_{k}", "both")

    # Compute error
    u_error = errornorm(u_true, u, norm_type = norm_type)
    return [float(h), float(u_error)]

def latexify_errornorm(error_norm):
    r"""Turns e.g. 'H1' into 'H^1(\Omega)' for Latex (in axis labels)."""
    return error_norm[0] + "^" + error_norm[1] + r"(\Omega)"

def main(args):
    # prep output folder
    time_now = dt_now()
    out_folder = init_outfolder(args.outputfolder + "/" + time_now)

    # prepare to store the error results
    h_ks = []
    u_errs = []

    # just so I can still use range(start, stop, step) below
    if args.step == 0:
        step = 1
    else:
        step = args.step

    # solve the problem for all given h and ks
    for N in range(args.N_min, args.N_max + 1, step):
        for k in args.k:
            h, u_err = define_and_solve(N, k, out_folder, args.figs, args.error_norm, args.bcs, args.symm_soln, args.regmesh)
            h_ks.append((h,k))
            u_errs.append(u_err)

    # Only create error figures if there are multiple h and one k.
    if (args.step > 0) and (len(args.k) == 1):
        grad_u = create_err_fig(h_ks, u_errs, out_folder, "$u$", "u", latexify_errornorm(args.error_norm), calc_slope = True)
    else:
        grad_u = None 

    # write and save a summary of the simulation.
    with open(f"{out_folder}/sim_output.txt", "w") as f:
        f.writelines([
            "Date & time: " + time_now + "\n",
            f"Params: {args.__str__().replace("Namespace", "")}" + "\n",
            "h_ks = " + str(h_ks) + "# pairs of (h,k) \n",
            "u_errs = " + str(u_errs) + "\n",
            "velocity convergence rate = " + str(grad_u) + "\n"
        ])

    done(out_folder)

    

if __name__ == "__main__":
    # Not all of the arguments used in mixed_stokes are used here, but a lot of them
    # are so we just use the same arg parser
    parser = init_parser(outfolder_default = "poisson_analyt_sims", k_default = 1)

    # Add in some extra arguments
    parser.add_argument("-error_norm", help = "Error norm to use, either 'H1' or 'L2'.", type = str, default = "H1")
    parser.add_argument("-bcs", help = "Boundary conditions to use: either 'dirichlet', 'mixed', or 'neumann'.", type = str, default = "dirichlet")
    parser.add_argument("--symm_soln", help = "Use example where solution is symmetric or not", action = BooleanOptionalAction, default = True)
    parser.add_argument ("--regmesh", help = "Whether to use a regular mesh (identical cells equally spaced) or an irregular mesh (spline).", action = BooleanOptionalAction, default = True)
    args = parser.parse_args()
    validate_args(args)
    assert args.bcs in ['dirichlet', 'mixed', 'neumann']

    main(args)