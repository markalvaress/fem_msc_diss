# example from https://www.wias-berlin.de/people/john/LEHRE/NUMERIK_IV_21_22/num_linear_saddle_prob_3.pdf p13
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import tripcolor
from firedrake.petsc import PETSc
import os

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

def define_and_solve(N: int) -> None:
    print(f"{N=}")

    # Define the mesh and function spaces
    mesh = UnitSquareMesh(N, N)
    x,y = SpatialCoordinate(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
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
        -2000*(-1 + x)**2 * (x**3 * (6 - 60*y**2) + y**2 * (-3 + 5*y**2) + x**4 * (-3 + 30*y**2) + x * (30*y**2 - 50*y**4) + 3*x**2 * (-1 - 5*y**2 + 25*y**4)),
        4000*(-1 + x)*y*(x**3 * (21 - 70*y**2) + 6*y**2 * (-1 + y**2) + x**4 * (-9 + 30*y**2) + x*(3 + 20*y**2 - 30*y**4) + 5*x**2 * (-3 + 4*y**2 + 6*y**4))
    ])

    grad_p = as_vector([
        pi**2*(y**3*cos(2*pi*x**2*y) - 2*x*y*(pi*x*y**2*cos(2*pi*x*y**2) + 2*pi*x*y**3*sin(2*pi*x**2*y) + sin(2*pi*x*y**2))),
        -pi**2*x*(-3*y**2*cos(2*pi*x**2*y) + x*(4*pi*x*y**2*cos(2*pi*x*y**2) + 2*pi*x*y**3*sin(2*pi*x**2*y) + sin(2*pi*x*y**2)))
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
    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

    # Solve the system
    solve(a == L, up, bcs=bcs, nullspace=nullspace)
    u, p = up.subfunctions

    # Plot the numerical and true results
    plot_and_save(u, p, f"stokes_figs/numerical_{N=}.png")
    plot_and_save(u_true, p_true, f"stokes_figs/exact_{N=}.png")

    print("Velocity error = ", errornorm(u_true, u))
    print("Pressure error = ", errornorm(p_true, p))

if __name__ == "__main__":
    if not os.path.exists("./stokes_figs"):
        os.mkdir("./stokes_figs")

    for k in range(1,5):
        define_and_solve(20*k)
