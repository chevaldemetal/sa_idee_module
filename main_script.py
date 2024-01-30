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
print("usage: ipython main_script.py [Power] [Nb CPU]")
try:
    Pow = int(sys.argv[1])
except IndexError:
    Pow = 4
try:
    nb_cpu = int(sys.argv[2])
except IndexError:
    nb_cpu = 1
                                                                         # --- macros -----------------------
"""
parameters and boundes to analyse

Capital (K)
        delta = 0.040000 [0.035000, 0.045000]
        nu = 3.000000 [2.610000, 3.390000]

Inflation (i)
        eta = 0.200000 [0.170000, 0.230000]
        mu = 1.700000 [1.666000, 1.734000]

Investment (I)
        kappa0 = 0.039700 [0.029775, 0.049625]
        kappa1 = 0.719000 [0.539250, 0.898750]

Productivity (p) 
        alpha = 0.010000 [0.007500, 0.012500]
        kaldor_verdoorn = 0.500000 [0.375000, 0.625000]

Dividends (D)
        div0 = 0.027500 [0.020625, 0.034375]
        div1 = 0.472900 [0.354675, 0.591125]

Population (N)
        deltanpop = 0.050000 [0.020000, 0.080000]
        npopbar = 5.040000 [4.662000, 5.418000]

Phillips (P)
        phi0 = -0.292000 [-0.293752, -0.290248]
        phi1 = 0.469000 [0.452585, 0.485415]
        gammaw = 0.500000 [0.450000, 0.550000]

Interest rate (r)
        phitaylor = 0.500000 [0.300000, 0.700000]
        istar = 0.020000 [0.012000, 0.028000]
        rstar = 0.020000 [0.012000, 0.028000]
        etar = 3.000000 [2.100000, 3.900000]

------------

Removed
        k_scale = 0.004500 [0.003352, 0.005647]
"""
names = [
    "delta",
    "nu",

    "eta",
    "mu",

    "kappa0",
    "kappa1",

    "alpha",
    "kaldor_verdoorn",

    "div0",
    "div1",

    "deltanpop",
    "npopbar",

    "phi0",
    "phi1",
    "gammaw",

    "phitaylor",
    "istar",
    "rstar",
    "etar",
]
groups = [
    "K",
    "K",

    "i",
    "i",

    "I",
    "I",

    "p",
    "p",

    "D",
    "D",

    "N",
    "N",

    "P",
    "P",
    "P",

    "r",
    "r",
    "r",
    "r",
]
bounds = [
    [0.035000, 0.045000],
    [2.610000, 3.390000],

    [0.170000, 0.230000],
    [1.666000, 1.734000],

    [0.029775, 0.049625],
    [0.539250, 0.898750],

    [0.007500, 0.012500],
    [0.375000, 0.625000],

    [0.020625, 0.034375],
    [0.354675, 0.591125],

    [0.020000, 0.080000],
    [4.662000, 5.418000],

    [-0.293752, -0.290248],
    [0.452585, 0.485415],
    [0.450000, 0.550000],

    [0.300000, 0.700000],
    [0.012000, 0.028000],
    [0.012000, 0.028000],
    [2.100000, 3.900000]
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
if nb_cpu==1:
    sa.run_IDEE(sa_class, path)
else:
    sa.run_IDEE_multiproc(sa_class, path, nb_cpu)

# 1.5 compute the outputs
if nb_cpu==1:
    resdata, bad_array, outputs_name = sa.make_outputs(path)
else:
    resdata, bad_array, outputs_name = sa.make_outputs_multiproc(path, nb_cpu)

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
