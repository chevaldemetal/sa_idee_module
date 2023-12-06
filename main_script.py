"""
file name: main_script.py
language: Python 3
date: 06/12/2023
author: Hugo A. Martin
email: martin_hugo@ymail.com
description: Main script of the code to test SA on IDEE.
"""
import sys
import sa_idee as sa
from sa_idee import plt
import os.path as road

POW = 1
                                                                         # 1. Functions to make data for SA
# 1.1 create a path
path = sa.check_dir()

# 1.2 initialize a class
sa_class = sa.make_sa_class(path, POW)
rep = input("\n  Number of samples is {:d}. Continue? Yes (Y) / No (N):\n".format(
    sa_class.samples.shape[0]))
if not rep=="Y":
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

# 3.2 plot the map
sa.plot_map(badc, goodc)

# 3.3 make the sensitivity analysis on the good simulations
sa_class = sa.run_SA(goodc)

# 3.4 plot it
sa_class.plot()
plt.tight_layout(**sa.PADS)
plt.savefig(road.join(path, "sa.pdf"))
plt.close("all")
