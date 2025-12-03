# Some little helper functions

from datetime import datetime 
import os
import functools
from firedrake import *

sim_outputs_folder = "./sim_outputs"

def dt_now():
    """Gets the current datetime as YYYY-MM-DD_HH-MM, for use in folder names."""
    now = str(datetime.now().replace(second=0, microsecond=0))[:-3].replace(" ", "_").replace(":", "-")
    return now

def init_outfolder(sim_name: str) -> str:
    """Creates the full output folder path, and creates the folders in the filesystem if they do not already exist."""
    out_folder = f"{sim_outputs_folder}/{sim_name}"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    return out_folder

def done(out_folder: str) -> None:
    print("Simulation complete.")
    print("Output saved to: " + out_folder)
    return

class FixAtPointBC(DirichletBC):
   r'''A special BC object for pinning a function at a point.

   :arg V: the :class:`.FunctionSpace` on which the boundary condition should be applied.
   :arg g: the boundary condition value.
   :arg bc_point: the point at which to pin the function.
       The location of the finite element DOF nearest to bc_point is actually used.
   '''
   def __init__(self, V, g, bc_point):
       super().__init__(V, g, bc_point)

   @functools.cached_property
   def nodes(self):
       V = self.function_space()

       point = [tuple(self.sub_domain)]
       vom = VertexOnlyMesh(V.mesh(), point)
       P0 = FunctionSpace(vom, "DG", 0)
       Fvom = Cofunction(P0.dual()).assign(1)

       # Take the basis function with the largest abs value at bc_point
       v = TestFunction(V)
       F = assemble(Interpolate(inner(v, v), Fvom))
       with F.dat.vec as Fvec:
           max_index, _ = Fvec.max()
       nodes = V.dof_dset.lgmap.applyInverse([max_index])
       nodes = nodes[nodes >= 0]
       return nodes