# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
import os
import utils
import scienceplots
import matplotlib
matplotlib.use('Agg')
plt.style.use("science")

out_folder = utils.init_outfolder("poisson_dirac_manual")

# Poisson's equation with dirichlet boundary conditions
A = np.diag([-2] * 6, k=3)
A += np.diag([-2] * 8, k=1)
A[2, 3] = 0
A[5, 6] = 0
A = A + A.T # The matrix is symmetric so copy the values from top-right to bottom-left
A += np.diag(8 * np.ones(9, dtype=int))

F = np.zeros(9)
F[0] = 1

U = np.linalg.solve(A, F)
U = U.reshape((3, 3))
# Adds the boundary nodes with value zero
U_padded = np.pad(U, 1, "constant", constant_values=0)

# plotting
fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(111, projection="3d")

pts = []
for i in range(5):
    for j in range(5):
        pts.append((i, j))

xs, ys = zip(*pts)

# Doing all this so we can see the correct triangles (cells) when plotting the results.
tri = Triangulation(
    xs,
    ys,
    [
        # bottom left triangles
        ## row 1
        (0, 1, 5),
        (0 + 1, 1 + 1, 5 + 1),
        (0 + 2, 1 + 2, 5 + 2),
        (0 + 3, 1 + 3, 5 + 3),
        ## row 2
        (5, 6, 10),
        (5 + 1, 6 + 1, 10 + 1),
        (5 + 2, 6 + 2, 10 + 2),
        (5 + 3, 6 + 3, 10 + 3),
        ## row 3
        (10, 11, 15),
        (10 + 1, 11 + 1, 15 + 1),
        (10 + 2, 11 + 2, 15 + 2),
        (10 + 3, 11 + 3, 15 + 3),
        ## row 4
        (15, 16, 20),
        (15 + 1, 16 + 1, 20 + 1),
        (15 + 2, 16 + 2, 20 + 2),
        (15 + 3, 16 + 3, 20 + 3),
        # Top left triangles
        ## row 1
        (1, 6, 5),
        (1 + 1, 6 + 1, 5 + 1),
        (1 + 2, 6 + 2, 5 + 2),
        (1 + 3, 6 + 3, 5 + 3),
        ## row 2
        (6, 11, 10),
        (6 + 1, 11 + 1, 10 + 1),
        (6 + 2, 11 + 2, 10 + 2),
        (6 + 3, 11 + 3, 10 + 3),
        ## row 3
        (11, 16, 15),
        (11 + 1, 16 + 1, 15 + 1),
        (11 + 2, 16 + 2, 15 + 2),
        (11 + 3, 16 + 3, 15 + 3),
        ## row 4
        (16, 21, 20),
        (16 + 1, 21 + 1, 20 + 1),
        (16 + 2, 21 + 2, 20 + 2),
        (16 + 3, 21 + 3, 20 + 3),
    ],
)

ax.plot_trisurf(tri, U_padded.flatten(), color = "coral", shade = False, edgecolor = "black")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u$")
# ax.set_xticks([0,1,2,3,4])
# ax.set_yticks([0,1,2,3,4])
# ax.set_zticks([0,0.1])
# ax.grid(markevery = 1, axis = "both")

# Can uncomment to print the solution vector
#print(U_padded)
fig.savefig(f"{out_folder}/soln_surface.png", dpi=300)
utils.done(out_folder)