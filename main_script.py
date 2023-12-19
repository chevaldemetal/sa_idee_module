"""
file name: main_script.py
language: Python 3
date: 06/12/2023
author: Hugo A. Martin
email: martin_hugo@ymail.com
description: Main script of the code to test SA on IDEE.
"""
                                                                         # --- imports ----------------------
import sys, os
import sa_idee as sa
import os.path as road
try:
    Pow = int(sys.argv[1])
except IndexError:
    Pow = 4
                                                                         # --- macros -----------------------
names = [
    "eta",
    "mu",
    "kappa0",
    "kappa1",
    "phi0",
    "phi1",
]
groups = [
    "Inflation",
    "Inflation",
    "Investment",
    "Investment",
    "Phillips",
    "Phillips"
]
bounds = [
    [0.180000, 0.220000],
    [1.674500, 1.725500],
    [0.029775, 0.049625],
    [0.539250, 0.898750],
    [-0.293168, -0.290832],
    [0.459620, 0.478380],
]
                                                                         # --- script -----------------------
                                                                         #
                                                                         # 1. Functions to make data for SA
# 1.1 create a path
path = sa.check_dir()

# 1.2 initialize a class
sa_class = sa.make_sa_class(Pow, names, groups, bounds)
rep = input("\n  Number of samples is {:d}. Continue? Yes (Y) / No (N):\n".format(
    sa_class.samples.shape[0]))
if not rep=="Y":
    os.system("rm -r {}".format(path))
    sys.exit("You chose to quit.")

# 1.3 save the class
sa.save_sa_class(sa_class, path)

# 1.4 make a set of simulations
sa.run_IDEE(sa_class, path)

# 1.5 compute the outputs
resdata, bad_array, outputs_name = sa.make_outputs(path)

# 1.6 set the results in the class
sa_class = sa.set_results(sa_class, resdata, bad_array, outputs_name)

# 1.7 save the class with the results
sa.save_sa_class(sa_class, path)
                                                                         # 2. Function to load data for SA
sa_class_load = sa.load_sa_class(path)
print("  Is classes equal?", sa.is_close(sa_class, sa_class_load))
                                                                         # 3. Perform the SA
# 3.1 sort the good and bas attractors
badc, goodc = sa.sort_attractors(sa_class)

# 3.2 make the sensitivity analysis on the good simulations
sa_class = sa.run_SA(goodc)
                                                                         # 4. Plots
# 4.1 plot the map
sa.plot_map(badc, goodc, path)

# 4.2 plot it
sa.plot_sa_class(sa_class, path)

# 4.3 plot histograms
sa.plot_histo(sa_class, path)

# 4.4 plot IDEE
file_name = "gemmes.out.World_default_set0"
sa.plot_IDEE(path, file_name)
