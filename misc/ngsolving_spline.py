from netgen.geom2d import SplineGeometry
from firedrake import *
from firedrake.pyplot import triplot
import matplotlib.pyplot as plt
import matplotlib
import sys
import scienceplots
matplotlib.use('Agg')
plt.style.use("science")

def make_spline_mesh():
    geo = SplineGeometry()

    geo.AddRectangle(
        p1 = (0,0),
        p2 = (1,1),
        bc = "rectangle",
        # leftdomain = 1,
        # rightdomain = 0
    )

    ngmsh = geo.GenerateMesh(maxh=0.1*(2**(1/2)))
    msh = Mesh(ngmsh) # make a firedrake mesh
    return msh

def save_reg_mesh():
    mesh = UnitSquareMesh(10, 10)
    return mesh

if __name__ == "__main__":
    try:
        if sys.argv[1] == "spline":
            mesh = make_spline_mesh()
        else:
            mesh = save_reg_mesh()
    except Exception as e:
        raise ValueError("Need to pass `spline' or `reg' as an argument.")
        
    fig, ax = plt.subplots()
    triplot(mesh, ax)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    if sys.argv[1] == "spline":
        ax.set_title("Spline mesh")
    else:
        ax.set_title("Regular mesh")
    fig.savefig(f"{sys.argv[1]}_mesh.png", dpi = 500)
    