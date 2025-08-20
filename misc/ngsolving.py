from netgen.meshing import *
m = Mesh(dim = 1)
N = 10
pnums = []

for i in range(0, N+1):
    pnums.append(m.add(MeshPoint(Pnt(i/N, 0, 0))))

idx = m.AddRegion("material", dim=1)
for i in range(N):
    m.add(Element1D([pnums[i], pnums[i+1]], index = idx))

idx_left = m.AddRegion("left", dim=0)
idx_right = m.AddRegion("right", dim=0)