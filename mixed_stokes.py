# example from https://www.wias-berlin.de/people/john/LEHRE/NUMERIK_IV_21_22/num_linear_saddle_prob_3.pdf p13
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import tripcolor
from firedrake.petsc import PETSc

mesh = UnitSquareMesh(32, 32)

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

u,p = TrialFunctions(Z)
v,q = TestFunctions(Z)

x,y = SpatialCoordinate(mesh)
f = Function(V).interpolate(as_vector([
    -4*y + 2*pi*cos(2*pi*x)*sin(2*pi*y),
    4*x + 2*pi*sin(2*pi*x)*cos(2*pi*y),
])) #10*exp(-(pow(x-0.5,2) + pow(y-0.5,2)) / 0.02)

a = (inner(grad(u), grad(v)) - inner(p, div(v)))*dx
L = inner(f, v)*dx

bcs = [DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3, 4))]

up = Function(Z)

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

solve(a == L, up, bcs=bcs, nullspace=nullspace,
    solver_parameters={"ksp_type": "gmres",
                        "mat_type": "aij",
                        "pc_type": "lu",
                        "pc_factor_mat_solver_type": "mumps"})
u, p = up.subfunctions

fig, axes = plt.subplots()
colors = tripcolor(u, axes=axes)
fig.colorbar(colors)
#plt.show() #doesn't work
plt.savefig("foobar.png")