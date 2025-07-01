import matplotlib.pyplot as plt
import sys

print(sys.argv)

with open("sim_output.txt") as f:
    lines = f.readlines()

# define h_ks, u_errs, p_errs
for i in [2,3,4]:
    exec(lines[i])

hs, ks = zip(*h_ks)
plt.plot(ks, u_errs, ".-")
plt.savefig("u_error")
plt.clf()
plt.plot(ks, p_errs, ".-")
plt.savefig("p_error")