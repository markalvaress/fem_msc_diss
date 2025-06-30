from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt

with open("sim_output_2025-06-27_11:14.txt") as f:
    lines = f.readlines()
    # assign h, u_errs, p_errs
    for i in [1,2,3]:
        exec(lines[i])
    h = np.array(h)
    u_errs = np.array(u_errs)
    p_errs = np.array(p_errs)

lr_results = linregress(np.log(h), np.log(u_errs + p_errs))
grad = lr_results.slope

plt.loglog(h, u_errs + p_errs)
plt.title(f"Slope = {grad:.2f}")
plt.savefig("joint_error.png")