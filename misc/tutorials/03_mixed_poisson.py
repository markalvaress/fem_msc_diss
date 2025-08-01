import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import tripcolor

mesh = UnitSquareMesh(32, 32)

BDM = FunctionSpace(mesh, "BDM", 1)
DG = FunctionSpace(mesh, "DG", 0)
W = BDM * DG

sigma, u = TrialFunctions(W) # this is what we're solving for
tau, v = TestFunctions(W) # these are our test functions that we integrate against

x,y = SpatialCoordinate(mesh)
f = Function(DG).interpolate(
    10*exp(-(pow(x-0.5,2) + pow(y-0.5,2)) / 0.02)
)

# define bilinear forms - the surface integral becomes 0 by the BC
a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
L = -f*v*dx

# apply explicit bcs to top and bottom of the vector function domain. 
# How am I meant to know that subdomain 3 and 4 refer to the right things?
bc0 = DirichletBC(W.sub(0), as_vector([0.0, -sin(5*x)]), 3)
bc1 = DirichletBC(W.sub(0), as_vector([0.0, sin(5*x)]), 4)

# Define a function to hold the solution (sigma, u)
w = Function(W)

solve(a == L, w, bcs = [bc0,bc1])
sigma, u = w.subfunctions

fig, axes = plt.subplots()
colors = tripcolor(u, axes=axes)
fig.colorbar(colors)
#plt.show() #doesn't work
plt.savefig("foo.png")