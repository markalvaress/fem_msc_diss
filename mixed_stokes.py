# example from https://www.wias-berlin.de/people/john/LEHRE/NUMERIK_IV_21_22/num_linear_saddle_prob_3.pdf p13
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import tripcolor
from firedrake.petsc import PETSc

# Define the mesh and function spaces
mesh = UnitSquareMesh(60, 60)

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

# Define trial (solution) and test functions
u,p = TrialFunctions(Z)
v,q = TestFunctions(Z)

# Define the RHS of the PDEs. We will add the two equations together to get our governing variational equation.
x,y = SpatialCoordinate(mesh)

lapl_u = as_vector([
    2000*(-1 + x**2)*(y**2*(3 - 8*y + 5*y**2) - 10*x*y**2*(3 - 8*y + 5*y**2) - 6*x**3*(1 - 8*y + 10*y**2) + 3*x**4*(1 - 8*y + 10*y**2) + 3*x**2*(1 - 8*y + 25*y**2 - 40*y**3 + 25*y**4)),
    -4000*(-1 + x)*y*(6*(-1 + y)**2*y**2 - 7*x**3*(3 - 12*y + 10*y**2) + x**4*(9 - 36*y + 30*y**2) + x*(-3 + 12*y - 40*y**2 + 60*y**3 - 30*y**4) + 5*x**2*(3 - 12*y + 16*y**2 - 12*y**3 + 6*y**4))
])

# grad_p = as_vector([
#     pi**2*(y**3*cos(2*pi*x**2*y) - 2*x*y*(pi*x*y**2*cos(2*pi*x*y**2) + 2*pi*x*y**3*sin(2*pi*x**2*y) + sin(2*pi*x*y**2))),
#     -pi**2*x*(-3*y**2*cos(2*pi*x**2*y) + x*(4*pi*x*y**2*cos(2*pi*x*y**2) + 2*pi*x*y**3*sin(2*pi*x**2*y) + sin(2*pi*x*y**2)))
# ])

# try constant pressure?
f = Function(V).interpolate(
    -lapl_u + as_vector([0,0])# grad_p
)

a = (inner(grad(u), grad(v)) - inner(p, div(v)) - inner(q, div(u)))*dx
L = inner(f, v)*dx

# the BC on the velocity space (first subspace of Z) is v = (0,0) on all boundaries
bcs = [DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3, 4))]

up = Function(Z)

# this was something to do with uniqueness of p. Maybe I don't need?
#nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# Solve the system
solve(a == L, up, bcs=bcs)#, nullspace=nullspace,
    # solver_parameters={"ksp_type": "gmres",
    #                     "mat_type": "aij",
    #                     "pc_type": "lu",
    #                     "pc_factor_mat_solver_type": "mumps"})
u, p = up.subfunctions

# Plot the results
fig, ax = plt.subplots(2, figsize = (10,10))
quiver(u, axes = ax[0])
ax[0].set_title("Velocity field")

colors = tripcolor(p, axes = ax[1])
fig.colorbar(colors)
ax[1].set_title("Pressure field")

plt.savefig("mixed_stokes_output.png")