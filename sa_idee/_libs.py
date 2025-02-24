"""
file name: _libs.py
language: Python 3
date: 12/12/2023
author: Hugo A. Martin
email: martin_hugo@ymail.com
description: libraries
"""
                                                                         # --- imports ----------------------
import sys, os, shutil
import os.path as road
import subprocess

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import linregress
from scipy.integrate import RK45
from scipy.sparse import diags

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.colors import CSS4_COLORS as COLORS
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
from cycler import cycler

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib import ProblemSpec

from timeit import default_timer as timer

import multiprocessing
from multiprocessing import Pool
                                                                         # --- macros -----------------------
                                                                         # macros for user
POW = 6
NAMES = [
    "eta",
    "mu",
    "kappa0",
]
GROUPS = [
    "Inflation",
    "Inflation",
    "Investment",
]
BOUNDS = [
    [0.16, 0.22],
    [2., 2.05],
    [0.03176, 0.04764],
]
OUTPUTS = [
    'amplitude',
    'debtratio',
    'dividends_ratio',
    'g0',
    'inflation',
    'kappa',
    'lambda',
    'main_frequency',
    'omega',
    'productivity_growth',
    'rb',
    'relaxation_time',
    'relaxation_time_inf',
    'relaxation_time_sup',
    'smallpi',
    'wage_growth'
]
PLOT_LIST = {
    'g0':'g',
    'lambda':'\lambda',
    'debtratio':'d',
    'omega':'\omega',
    'wage_growth':'\dot{w}/w',
    'productivity_growth':'\dot{a}/a',
    'smallpi':'\pi',
    'kappa':'\kappa',
    'dividends_ratio':'\Delta',
    'inflation':'i',
    'rb':'r',
    'npop':'N',
    'capital':'K',
}
NUMCOLS = 3
LAST_YEAR = 3000
TMAX = 3000
                                                                         # macros for devs
RAISE = True
                                                                         # 1 capital 18 omega 19 lambda
COM_IDEE = "./gemmes"
XMP_FILE = "gemmes.dat.example"
DAT_FILE = "gemmes.dat.World_default"
OUT_FILE = "gemmes.out.World_default"
REG_FILE = "regions.dat"
DIR_IDEE = "/home/admin/Documents/sciences/postdocGU/codes/GEMMESCLIM/gemmes_cpl/sources/"
RAW_PATH = "raw_data"
DIR_LOC = os.getcwd()
DIR_SAVE = road.join(DIR_LOC, "outputs_")
DT = 1./12
INFTY_SMALL = 1.E-12
POURCENT = 0.01
WINDOW_FREQ = 200
WINDOW_AMP = [-400, 0]
WINDOW_MEAN = [-100, 0]
                                                                         # --- macros -----------------------
A4W = 8.27
rcParams.update({'font.size': 8, 'lines.linewidth': 1})
PADS = {
    "pad":0.,
    "w_pad":0.,
    "h_pad":0.
}
COLORS = [
    COLORS["black"],
    COLORS["dimgray"],
    COLORS["gray"],
    COLORS["darkgray"],
]
LS = ['-', '--', ':', '-.']
cc = []
ls = []
for j in LS:
    for i in COLORS:
        cc.append(i)
        ls.append(j)
plt.rc('axes',
    prop_cycle=(cycler(color=cc) +
                cycler(linestyle=ls))
)
