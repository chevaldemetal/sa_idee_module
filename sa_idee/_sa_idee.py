"""
file name: sa_idee.py
language: Python 3
date: 20/11/2023
author: Hugo A. Martin
email: martin_hugo@ymail.com
description: This module is can be used to run the sensitivity analysis of the model IDEE.

TODO:

- negative S1
- - faire un histogramme et enlever les valeurs aberrantes
- - relaxation time très très grands#
- - relaxation time très très grands
- - compare S1 and histograms
- - remove extreme values in outputs
- - test the centered method https://github.com/SALib/SALib/issues/109
- - increase the sample
- - test other methods than Sobol's one
"""
                                                                         #--- imports -----------------------
import sys, os, shutil
import subprocess
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib import ProblemSpec
from timeit import default_timer as timer
from scipy.signal import hilbert
from scipy.stats import linregress
import os.path as road
from cycler import cycler
from matplotlib.colors import CSS4_COLORS as COLORS
rcParams.update({'font.size': 8, 'lines.linewidth': 1})
PADS = {
    "pad":0.,
    "w_pad":0.,
    "h_pad":0.
}
                                                                         #--- macros for users --------------
POW = 6
NAMES = [
    "eta",
    "mu",
    "kappa0",
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
    'capital':'K',
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
}
NUMCOLS = 3
LAST_YEAR = 3000
TMAX = 3000
YMIN, YMAX = -0.1, 1.1
                                                                         #--- macros for devs ---------------
RAISE = True
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
                                                                         # 1 capital 18 omega 19 lambda
INDICES = [1, 18, 19]
COM_IDEE = "./gemmes"
XMP_FILE = "gemmes.dat.example"
DAT_FILE = "gemmes.dat.World_default"
OUT_FILE = "gemmes.out.World_default"
DIR_IDEE = "/home/admin/Documents/sciences/postdocGU/codes/GEMMESCLIM/gemmes_cpl/sources/"
RAW_PATH = "raw_data"
DIR_LOC = os.getcwd()
DIR_SAVE = road.join(DIR_LOC, "outputs_")
DT = 1./12
INFTY_SMALL = 1.E-12
POURCENT = 0.01
WINDOW_FREQ = 100
WINDOW_AMP = [-400, 0]
WINDOW_MEAN = [-100, 0]
A4W = 8.27
                                                                         #--- functions ---------------------
def check_bad_attractor(data):
    """
    This function checks whether the simulation converges toward the bad attractor.

    Input
        data : numpy.ndarray (float)
            the array containing the simulation's data
    Output
        is_bad : boolean
            True wheter it converges toard the bad attractor
    """
    is_bad = False
    for ind in INDICES:
        raw = data[:, ind]
        is_bad = is_bad or (raw<POURCENT).any() or np.isnan(raw).any()

    return is_bad

def check_dir():
    """
    Check wether the outputs directory exists.

    Output
        name_dir : string
            the name of the directory that constains the outputs
    """
    it_exists = True
    num = 0
    while it_exists:
        num += 1
        name_dir = DIR_SAVE + str(num)
        it_exists = road.exists(name_dir)

    os.mkdir(name_dir)

    return name_dir

def comp_growth(data, nb):
    """
    Computes the growth rate of the array data.

    Input
        data : numpy.ndarray
            the raw data
        nb : integer
            the window length
    Output
        growth : numpy.ndarray
            the growth rate (same size filled with zeros)
    """
    g = (np.roll(data, -1) - data)/DT/data
    growth = np.convolve(g[::-1], np.ones(nb), "valid") / nb
    growth[0] = np.nan
    growth = np.hstack(([np.nan]*(nb-1), growth))[::-1]

    return growth

def comp_main_freq(signal):
    """
    Computes the main frequency of the signal.

    Input
        signal : numpy.ndarray
            the signal
    Output
        main_freq : float
            the main frequency
        sp : numpy.ndarray
            sample of powers
        freq : numpy.ndarray
            frequencies
    """
    sp = np.fft.rfft(signal)
    nf = np.argmax(np.absolute(sp)[1:])
    freq = np.fft.rfftfreq(signal.size, d=DT)[1:]
    main_freq = freq[nf]

    return main_freq, sp, freq

def comp_mean(raw, selectamp):
    """
    This function can be used to compute the mean of the signal.

    Input
        raw : numpy.ndarray (float)
            the data
        selectamp : numpy.ndarray (boolean)
            the selected window
    Output
        mean : float
            the mean
        damp : float
            the signal amplitude
    """
                                                                         # compute the main amplitude
    damp = 0.5*(np.nanmax(raw[selectamp]) - np.nanmin(raw[selectamp]))
                                                                         # compute the mean
    mean = np.nanmin(raw[selectamp]) + damp

    return mean, damp

def comp_nb_samples(c):
    """
    Compute the number of samples a sa_class has.

    Input
        c : salib.util.problem.problemspec
            class of the library salib
    Ouput
        c : salib.util.problem.problemspec
            class updated with nb_samples
    """
    try:
        n = c.samples.shape[0]
    except Exception:
        n = 0
    c["nb_samples"] = n

    return c

def comp_observations(time, raw):
    """
    Compute the observations on the omega variable.

    Input
        time : numpy.ndarray (float)
            time
        raw : numpy.ndarray (float)
            the raw signal (most of the time 'omega')
    Output
        outs : dict (*)
            mean : (float)
                mean at the end
            main_freq : (float)
                main frequency
            relax_time : (float)
                relaxation time
            damp : (float)
                amplitude
        buffs : dict (*)
            delay : (float)
                delay between time[0] and time[argmax(raw)]
            xi : numpy.ndarray (float)
                lower envelope
            xs : numpy.ndarray (float)
                upper envelope
            ress : list (numpy.ndarray (float))
                linear regressions of envelopes
            relax_times : list (float)
                relaxation times of envelopes
            label : string
                label of the chosen envelope
            status : string
                indicates converge, diverge, trans_inf, trans_sup
            initial_times : list (float)
                the times from which it converges
    """
                                                                         # selects
    selectf = time <= (time[0] + WINDOW_FREQ)
    selectamp = (time >= (time[-1]+WINDOW_AMP[0])) * \
        (time <= (time[-1]+WINDOW_AMP[1]))
                                                                         # compute the amplitude and mean
    mean, damp = comp_mean(raw, selectamp)
                                                                         # compute the main frequency
    main_freq, sp, f = comp_main_freq(raw[selectf])
    Tmax = time[selectf]
    Tmax = Tmax[np.argmax(raw[selectf])]
    delay = Tmax - time[0]

                                                                         # compute the relaxation time
    gradient = np.gradient(
        raw + 0.01*np.sin(2.*np.pi*main_freq*(time-delay)),
        DT
    )
    gradient[np.nanargmax(raw)] = 0.
    gradient[np.nanargmin(raw)] = 0.
                                                                         # check whether its converges or not
    anal_sup = hilbert(raw - mean)
    big_env_sup = np.abs(anal_sup)
    tmp_sel = time <= (time[0]+100)
    nanmax_ini, nanmin_ini = np.nanmax(raw[tmp_sel]), np.nanmin(raw[tmp_sel])
    amp_ini = nanmax_ini - nanmin_ini

    tmp_sel = ((time[0]+300) <= time) * (time <= (time[0]+400))
    mean_sup_infty = np.nanmean(big_env_sup[tmp_sel])
    nanmax_infty, nanmin_infty = np.nanmax(raw[tmp_sel]), np.nanmin(raw[tmp_sel])
    amp_infty = nanmax_infty - nanmin_infty

    if (amp_ini >= amp_infty) and (nanmax_ini >= nanmax_infty) and (nanmin_ini <= nanmin_infty):
        status = "converge"
    elif (amp_ini <= amp_infty) and (nanmax_ini <= nanmax_infty) and (nanmin_ini >= nanmin_infty):
        status = "diverge"
    elif (nanmax_ini >= nanmax_infty) and (nanmin_ini >= nanmin_infty):
        status = "trans_inf"
    elif (nanmax_ini <= nanmax_infty) and (nanmin_ini <= nanmin_infty):
        status = "trans_sup"
    else:
        plt.close('all')
        raise ValueError("No status has been found.")

    selpos = ((raw - mean) >= 0.)
    selneg = ((raw - mean) <= 0.)
    changing_signs = (np.roll(gradient, -1) * gradient)<=0.
    sel_sup = changing_signs * selpos
    sel_inf = changing_signs * selneg

    tinf = time[sel_inf]
    rawinf = raw[sel_inf]

    if status=="converge" or status=="trans_sup":
        sel_inf_2 = np.logical_or(
            ((np.roll(rawinf, -1) - rawinf)>=0.),
            np.abs(rawinf-mean)<0.1
        )
        sel_inf_2[0] = rawinf[0] <= rawinf[1]
    elif status=="diverge" or status=="trans_inf":
        sel_inf_2 = [True]*tinf.size

    tsup = time[sel_sup]
    rawsup = raw[sel_sup]

    if status=="converge" or status=="trans_inf":
        sel_sup_2 = np.logical_or(
            ((np.roll(rawsup, -1) - rawsup)<=0.),
            np.abs(rawsup-mean)<0.1
        )
        sel_sup_2[0] = rawsup[0] >= rawsup[1]
    elif status=="diverge" or status=="trans_sup":
        sel_sup_2 = [True]*tsup.size

    rawinf = rawinf[sel_inf_2]
    tinf = tinf[sel_inf_2]

    rawsup = rawsup[sel_sup_2]
    tsup = tsup[sel_sup_2]

    xi = np.interp(time, tinf, rawinf)
    xs = np.interp(time, tsup, rawsup)

    xi = np.fmin(xi, mean)
    xs = np.fmax(xs, mean)

    relax_times = []
    envs = [xi, xs]
    ress = []
    raws = [rawinf, rawsup]
    labels = ["inf", "sup"]
    initial_times = []
    for ii, amp_env in enumerate(envs):
        notnan = ~np.isnan(amp_env)
        tmpraw = raws[ii]

        nanmax, nanmin = np.nanmax(amp_env), np.nanmin(amp_env)
                                                                         # here we define the sign of the
                                                                         # desired slope
        if ii==0:
            if status=="converge" or status=="trans_sup":
                y = nanmax
                scale = 1.
            elif status=="diverge" or status=="trans_inf":
                y = nanmin
                scale = -1.
        else:
            if status=="converge" or status=="trans_inf":
                y = nanmin
                scale = -1.
            elif status=="diverge" or status=="trans_sup":
                y = nanmax
                scale = 1.

        Delta = np.gradient(amp_env, DT)
        Delta[np.logical_not(np.sign(Delta)==scale)] = 0.

        argD = np.nanargmax(np.abs(Delta))
        slope = Delta[argD]
        not_zero = np.abs(Delta)>INFTY_SMALL

        if (np.count_nonzero(not_zero)==0) or (slope==0.):
            tmpraw = [amp_env[0]]
            ts = [time[-1]]
            slope = 1.E-15
        else:
            tmpraw = amp_env[not_zero]
            ts = time[not_zero]
        intercept = tmpraw[0] - slope*ts[0]
        res = {"slope":slope, "intercept":intercept}

        tmp = res["intercept"] + res["slope"]*time

        t = (y - res["intercept"]) / res["slope"] - ts[0]
        relax_times.append(t)
        initial_times.append(ts[0])
        ress.append(res)

    label = labels[np.argmin(relax_times)]
    relax_time = np.min(relax_times)

    outs = {
        "mean":mean,
        "main_freq":main_freq,
        "relax_time":relax_time,
        "damp":damp
    }
    buffs = {
        "delay":delay,
        "xi":xi,
        "xs":xs,
        "ress":ress,
        "relax_times":relax_times,
        "label":label,
        "status":status,
        "initial_times":initial_times
    }

    return outs, buffs

def empty_print(a):
    print(a)
    return 1

def extend_IDEE(lvars, data):
    """
    This function adds new fields compputed from the raw data.

    Input
        lvars : list (string)
            the list of variables names
        data : numpy.ndarray (float)
            the raw data
    Output
        lvars : list (float)
            the extended list of variables
        data : numpy.ndarray (float)
            the extended data
    """
    nb = int(1./DT)
    nbrows = data.shape[0]
                                                                         # append investment ratio kappa
    lvars.append("kappa")
    new = np.zeros((nbrows,1))
    new[:,0] = data[:,36] / data[:,22]
    data = np.hstack((data, new))
                                                                         # append dividends ratio
    lvars.append("dividends_ratio")
    new = np.zeros((nbrows,1))
    new[:,0] = data[:,27] - data[:,37] / data[:,22] / data[:,6]
    data = np.hstack((data, new))
                                                                         # append wage growth rate
    lvars.append("wage_growth")
    new = np.zeros((nbrows,1))
    new[:,0] = comp_growth(data[:,4], nb)
    data = np.hstack((data, new))
                                                                         # append productivity growth
    lvars.append("productivity_growth")
    new = np.zeros((nbrows,1))
    new[:,0] = comp_growth(data[:,5], nb)
    data = np.hstack((data, new))
                                                                         # append population growth
    lvars.append("population_growth")
    new = np.zeros((nbrows,1))
    new[:,0] = comp_growth(data[:,2], nb)
    data = np.hstack((data, new))
    return lvars, data

def IDEE(params, n, name_dir):
    """
    This function defines a set of params, sets a single problem to simulate and runs it.

    Input
        params : dict (float)
            the set of parameters to change
        n : integer
            the number associated to this specific set of params
        name_dir : string
            the outputs' directory
    """
    os.chdir(DIR_IDEE)
                                                                         # change the file of parameters
    shutil.copyfile(XMP_FILE, DAT_FILE)
    f = open(DAT_FILE, 'a')
    f.write(" region%dt={:.6f}\n".format(DT))
    f.write(" region%Tmax={:.6f}\n".format(TMAX))
    for k,v in params.items():
        f.write(" region%{}={:.6f}\n".format(k,v))
    f.write('\n/\n')
    f.close()
                                                                         # IDEE is ran here
    subprocess.run(COM_IDEE)
                                                                         # save the outputs
    shutil.copyfile(OUT_FILE, road.join(name_dir, OUT_FILE + '_set{:d}'.format(n)))

    os.chdir(DIR_LOC)

def is_close(c1, c2):
    """
    Returns True whether the classes sa_c1 and sa_c2 are similar.

    Input
        c1 : SALib.util.problem.ProblemSpec
            first class of the library SAlib
        c2 : SALib.util.problem.ProblemSpec
            second class of the library SAlib
    Output
        is_close : boolean
            True whether the two classes are equal (with a controled error)
    """
    error = lambda b, k: empty_print(" {} are different".format(k)) if (not b) else 0
                                                                         # list of cheks
                                                                         # num_vars
    k = "num_vars"
    is_close = error(c1[k]==c2[k], k)
                                                                         # names and outputs
    for k in ["names", "outputs"]:
        n1 = c1[k]
        n2 = c2[k]
        for n, name in enumerate(n1):
            is_close += error(name==n2[n], k)
                                                                         # bounds
    k = "bounds"
    is_close += error(np.allclose(c1[k], c2[k]), k)
                                                                         # results
    try:
        s1 = c1.results.size
        s2 = c2.results.size
        k = "results array"
        is_close += error(s1==s2, k)
        is_close += error(np.allclose(c1.results, c2.results), k)

    except AttributeError:
        print("At least one class does not have any results array.")

    try:
        ba1 = c1["bad_array"]
        ba2 = c2["bad_array"]
        k = "results array"
        is_close += error(ba1.size==ba2.size, k)
        is_close += error(np.allclose(ba1, ba2), k)

    except KeyError:
        print("At least one class does not have any bad_array.")

    return is_close==0

def make_outputs(path):
    """
    This function loads a sample of simulations and make the result array.

    Input
        path : string
            location of the simulations
    Output
        result : numpy.ndarray (float)
            the array of resulted post-treatment data
        bad_array : numpy.ndarray (boolean)
            True if it is a simulation toward the bad attractor
        keys : list (string)
            outputs names
    """
    name_dir = road.join(path, RAW_PATH)
    samples = np.loadtxt(road.join(path, "sample.dat"))
    nums = samples.shape[0]
    bad_array = np.zeros(nums, dtype=np.bool8)
    nb_len = len(OUTPUTS)
    keys = sorted(list(OUTPUTS))
    output_list = OUTPUTS.copy()
    output_list.remove('amplitude')
    output_list.remove('main_frequency')
    output_list.remove('relaxation_time')
    output_list.remove('relaxation_time_inf')
    output_list.remove('relaxation_time_sup')

    results = np.ones((nums, nb_len))
                                                                         # check whether there is an existing file
    tmp_file_name = road.join(name_dir, "outputs_in_process.dat")
    tmp_bad_name = road.join(name_dir, "badarray_in_process.dat")
    if road.exists(tmp_file_name) and road.exists(tmp_bad_name):
        print("Loading {}".format(tmp_file_name))
        print("Loading {}".format(tmp_bad_name))
        results = np.loadtxt(fname=tmp_file_name)
        bad_array = np.loadtxt(tmp_bad_name).astype(np.bool8)
        tmp = np.sum(results, axis=1)==nb_len
        if tmp.any():
            first_ind = np.argmax(tmp)
            print("\nMake outputs from {:d}".format(first_ind))
        else:
            first_ind = tmp.size
            print("\nAll results are already computed.")
    else:
        print("No {} found.\nStart from 0.".format(tmp_file_name + " " + tmp_bad_name))
        first_ind = 0

    print("  {:d} simulations to do".format(nums-first_ind))
    for n in range(first_ind, nums):
                                                                         # load here
        file_name = road.join(name_dir, OUT_FILE+'_set{:d}'.format(n))
        data = np.loadtxt(file_name, skiprows=1)
                                                                         # check whether it is bad attracted
        bad_array[n] = check_bad_attractor(data)
        sample = [n] + samples[n,:].tolist()
                                                                         # load variable names
        f = open(file_name, "r")
        lvars = f.readline()
        f.close()
        lvars = lvars.split('  ')[:-1]
        lvars = lvars[0].split(' ') + lvars[1:]
                                                                         # extend data
        lvars, data = extend_IDEE(lvars, data)
        time = data[:,0]
        selectamp = (time >= (time[-1]+WINDOW_AMP[0])) * (time <= (time[-1]+WINDOW_AMP[1]))

        outputs = {}
        if not bad_array[n]:
            for var in output_list:
                raw = 100.*data[:, lvars.index(var)]
                if var=="omega":
                    outs, buffs = comp_observations(time, raw)
                mean, damp = comp_mean(raw, selectamp)
                outputs[var] = mean

            outputs["amplitude"] = outs["damp"]
            outputs["main_frequency"] = outs["main_freq"]
            outputs["relaxation_time"] = outs["relax_time"]
            outputs["relaxation_time_inf"] = buffs["relax_times"][0]
            outputs["relaxation_time_sup"] = buffs["relax_times"][1]

        else:
            for var in OUTPUTS:
                outputs[var] = 0.

            outputs["amplitude"] = 0.
            outputs["main_frequency"] = 0.
            outputs["relaxation_time"] = 0.
            outputs["relaxation_time_inf"] = 0.
            outputs["relaxation_time_sup"] = 0.

        line = []
        for key in keys:
            line.append(outputs[key])
        results[n, :] = line
                                                                         # save the temporary file all 10th
        if (n%10==0) or (n==nums-1):
            print('   {:d}'.format(n))
            np.savetxt(fname=tmp_file_name, X=results)
            np.savetxt(fname=tmp_bad_name, X=bad_array)

    return results, bad_array, keys

def make_sa_class(Pow=2, names=NAMES, bounds=BOUNDS):
    """
    This function makes a class of the library SAlib with the required parameters.

    Input
        Pow : integer
            the power used to determine the sample size
        names : list (string)
            name of variables in the SA
        bounds : list (integer)
            bounds of the variables
    Ouput
        sa_class : SALib.util.problem.ProblemSpec
            the class of the library SAlib
    """
    sa_class = ProblemSpec({
        "num_vars":len(names),
        "names":names,
        "bounds":bounds,
        "outputs":OUTPUTS,
    })
                                                                         # get the parameters
    sa_class.sample_saltelli(N=2**Pow)
    print("  sa_class made with {:d} samples".format(
        sa_class.samples.shape[0]
    ))

    return sa_class

def load_sa_class(path):
    """
    This functions loads the data and make a sa_class.

    Input
        path : string
            the path where to load the data
    Output
        sa_class : SALib.util.problem.ProblemSpec
            the class of the library SAlib
    """
    filename = road.join(path, "sample.dat")
    sample = np.loadtxt(filename)
    f = open(filename)
    names = f.readline().split(' ')[1:-1]
    f.close()

    filename = road.join(path, "bounds.dat")
    bounds = np.loadtxt(filename)

    sa_class = ProblemSpec({
        "num_vars":len(names),
        "names":names,
        "bounds":bounds.tolist(),
        "outputs":["None"]
    })
    sa_class.set_samples(sample)
    sa_class = comp_nb_samples(sa_class)

    filename = road.join(path, "outputs_to_analyze.dat")
    if road.exists(filename):
        results = np.loadtxt(filename)
        f = open(filename)
        out_names = f.readline().split(' ')[1:-1]
        f.close()
        sa_class["outputs"] = out_names
        sa_class.set_results(results)
    else:
        sa_class["outputs"] = OUTPUTS
        print("Outputs cannot be found. Defaults outputs are assigned.")
        print(OUTPUTS)

    filename = road.join(path, "bad_array.dat")
    if road.exists(filename):
        bad_array = np.loadtxt(filename).astype(np.bool8)
        sa_class["bad_array"] = bad_array.copy()

    return sa_class

def perso_savefig(fig, path, figure_name, show):
    """
    Personal function for saving figures.

    Input
        fig : matplotlib.pyplot.Figure
            figure
        path : string
            path of data
        figure_name : string
            name of figure
        show : boolean
            if True show, else save figure
    """
    if not show:
        plt.tight_layout(**PADS)
        plt.savefig(road.join(path, figure_name),
            bbox_inches="tight", pad_inches=0.1,
        )
        print("save figure {}".format(road.join(path, figure_name)))
        plt.close(fig)
    else:
        plt.show()
    plt.close(fig)

def plot_histo(c, path, n_bins=100, figsize=(15,10), figure_name="histograms.pdf", show=False):
    """
    Plot histograms of results.

    Input
        c : SALib.util.problem.ProblemSpec
            class of the library SAlib
        path : string
            path of the data
        n_bins : integer
            number of bins
        perc : list (float)
            percentiles for extreme values removal
        figsize : tuple (float)
            figure size
        figure_name : string
            file name
        show : boolean
            if True, show else save figure
    """
    nbrows, nbcols = 4, 4
    try:
        s = c.results.shape[0]
    except AttributeError:
        print("Cannot draw histograms without results.")
        pass
                                                                         # plot histograms
    fig, axes = plt.subplots(nbrows, nbcols, figsize=figsize)
    for n, out in enumerate(c["outputs"]):
        i, j = n//nbcols, n%nbcols
        raw = c.results[:, n]
        ax = axes[i, j]
                                                                         # postdata process
        std = np.nanstd(raw)
        mean = np.nanmean(raw)
                                                                         # plots
        ax.hist(raw, bins=n_bins)
        ylims = ax.get_ylim()
        line, = ax.plot([mean-std, mean-std], [0., ylims[-1]], linestyle="--")
        ax.plot(
            [mean+std, mean+std],
            [0., ylims[-1]],
            color=line.get_color(),
            linestyle=line.get_linestyle()
        )
        ax.set_ylabel(out)
        #ax.set_ylim(bottom=0.1*s)

    perso_savefig(fig, path, figure_name, show)

def plot_main_details_IDEE(time, raw):
    """
    Return a figure that shows the main detail of the computed quantities
    for the SA of IDEE.

    Input
        time : numpy.ndarray (float)
            time
        raw : numpy.ndarray (float)
            the raw signal (most of the time 'omega')
    Output
        figlam : matplotlib.pyplot.figure
            figure with all details
        axlam : matplotlib.axes.Axes
            left figure with details
        axlam1 : matplotlib.axes.Axes
            right figure with linear regressions of envelopes
    """
    figlam = plt.figure(figsize=(A4W, 3.5))
    gs = GridSpec(
        nrows=1,
        ncols=2,
        width_ratios=[2, 1],
        wspace=0.1,
    )
    axlam = figlam.add_subplot(gs[:, :-1])
    axlam1 = figlam.add_subplot(gs[0,-1])
                                                                         # compute other observation quantities
    outs, buffs = comp_observations(time, raw)

    mean = outs["mean"]
    main_freq = outs["main_freq"]
    relax_time = outs["relax_time"]
    damp = outs["damp"]
    delay = buffs["delay"]
    xi = buffs["xi"]
    xs = buffs["xs"]
    ress = buffs["ress"]
    relax_times = buffs["relax_times"]
    label = buffs["label"]
    status = buffs["status"]
    initial_times = buffs["initial_times"]

    t0 = initial_times[0] if label=="inf" else initial_times[1]
    labels = ["inf", "sup"]
    selectr = time <= (time[0] + WINDOW_FREQ)
                                                                         #
                                                                         # plots
    st = time<=LAST_YEAR
    timest = time[st]
    sst = timest.size
                                                                         # plot raw data
    axlam.plot(timest, raw[st], label="raw_data")
                                                                         # plot mean
    axlam.plot(timest, mean*np.ones(sst))
                                                                         # plot relaxation time
    hr = 0.5*np.amax(np.abs(raw-mean)[selectr])
    line, = axlam.plot([t0+relax_time]*2, [mean-hr, mean+hr])
    for ii in range(2):
        axlam.plot(
            [t0+(2+ii)*relax_time]*2,
            [mean-hr, mean+hr],
            color=line.get_color(),
            linestyle=line.get_linestyle()
        )
                                                                         # plot approximated signal
    approx = mean + damp*np.sin(2.*np.pi*main_freq*(timest-delay))
    axlam.plot(timest, approx, label=r"$M + (A/2) \sin ( 2 \pi f t )$")
                                                                         # plot envelope
    line, = axlam.plot(timest, xs[st], label="$a(t)$ envelope")
    axlam.plot(timest, xi[st],
        color=line.get_color(),
        linestyle=line.get_linestyle()
    )
                                                                         # plots in axlam1
                                                                         # inf
    tmp = ress[0]["intercept"] + ress[0]["slope"]*timest
    nanminxi = np.nanmin(xi)
    nanmaxxi = np.nanmax(xi)
    if status=="converge" or status=="trans_sup":
        tmpsel = tmp <= nanmaxxi
        toplot = nanmaxxi
    elif status=="diverge" or status=="trans_inf":
        tmpsel = tmp >= nanminxi
        toplot = nanminxi
    tmpsel = tmpsel * (timest >= initial_times[0])
    lineres, = axlam1.plot(timest[tmpsel], tmp[tmpsel],
        color="C2"
    )
    axlam1.plot(timest, toplot*np.ones(sst),
        color=lineres.get_color(),
        linestyle=lineres.get_linestyle(),
    )
    axlam1.plot(timest[tmpsel], toplot*np.ones(np.count_nonzero(tmpsel)),
        color=lineres.get_color(),
        linestyle=lineres.get_linestyle(),
        linewidth=2,
    )
    axlam1.plot(timest, xi[st],
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            label=r"$a(t)$"
    )
    axlam1.plot(
        [initial_times[0]]*2,
        [nanminxi, nanmaxxi],
        color=lineres.get_color(),
        linestyle=lineres.get_linestyle(),
    )
                                                                         # sup
    tmp = ress[1]["intercept"] + ress[1]["slope"]*timest
    nanminxs = np.nanmin(xs)
    nanmaxxs = np.nanmax(xs)
    if status=="converge" or status=="trans_inf":
        tmpsel = tmp >= nanminxs
        toplot = nanminxs
    elif status=="diverge" or status=="trans_sup":
        tmpsel = tmp <= nanmaxxs
        toplot = nanmaxxs
    tmpsel = tmpsel * (timest >= initial_times[1])
    axlam1.plot(timest[tmpsel], tmp[tmpsel],
        color=lineres.get_color(),
        linestyle=lineres.get_linestyle(),
    )
    axlam1.plot(timest, toplot*np.ones(sst),
        color=lineres.get_color(),
        linestyle=lineres.get_linestyle(),
    )
    axlam1.plot(timest[tmpsel], toplot*np.ones(np.count_nonzero(tmpsel)),
        color=lineres.get_color(),
        linestyle=lineres.get_linestyle(),
        linewidth=2,
    )
    axlam1.plot(timest, xs[st],
            color=line.get_color(),
            linestyle=line.get_linestyle(),
    )
    axlam1.plot(
        [initial_times[1]]*2,
        [nanminxs, nanmaxxs],
        color=lineres.get_color(),
        linestyle=lineres.get_linestyle(),
    )
                                                                         # plot amplitude
    line, = axlam.plot(
        [time[-1]+WINDOW_AMP[0], time[-1]+WINDOW_AMP[1]],
        [mean+damp, mean+damp],
    )
    axlam.plot(
        [time[-1]+WINDOW_AMP[0], time[-1]+WINDOW_AMP[1]],
        [mean-damp, mean-damp],
        color=line.get_color(),
        linestyle=line.get_linestyle()
    )
                                                                         # plot text relaxation time
    props = dict(boxstyle='round', facecolor='white', alpha=1.)
    line, = axlam.plot(
        [t0+relax_time, t0+1.5*relax_time],
        [mean+hr, mean+1.2*hr]
    )
    axlam.plot(
        [t0+1.5*relax_time, t0+2.*relax_time],
        [mean+1.2*hr, mean+hr],
        color=line.get_color(),
        linestyle=line.get_linestyle()
    )
    axlam.text(
        t0+1.5*relax_time,
        mean+1.2*hr,
        r'$\tau = \min(\tau_{inf}, \tau_{sup}) = %.1f$ y' % relax_time,
        horizontalalignment='center',
        verticalalignment='center',
        bbox=props
    )
                                                                         # plot text main frequency
    axlam.plot(
        [time[0]+1/main_freq+delay, time[0]+1.5/main_freq+delay],
        [mean+damp, mean+1.25*damp+1.],
        color=line.get_color(),
        linestyle=line.get_linestyle()
    )
    axlam.plot(
        [time[0]+1.5/main_freq+delay, time[0]+2/main_freq+delay],
        [mean+1.25*damp+1., mean+damp],
        color=line.get_color(),
        linestyle=line.get_linestyle()
    )
    axlam.text(
        time[0]+1.5/main_freq+delay,
        mean+1.25*damp + 1.,
        r'$f^{-1} = %.1f$ y' % (1./main_freq),
        horizontalalignment='center',
        verticalalignment='center',
        bbox=props
    )
                                                                         # plot text amplitude
    axlam.plot(
        [time[-1]+0.5*(WINDOW_AMP[0]+WINDOW_AMP[1])]*2,
        [mean+damp, mean],
        color=line.get_color(),
        linestyle=line.get_linestyle()
    )
    axlam.plot(
        [time[-1]+0.5*(WINDOW_AMP[0]+WINDOW_AMP[1])]*2,
        [mean, mean-damp],
        color=line.get_color(),
        linestyle=line.get_linestyle()
    )
    axlam.text(
        time[-1]+0.5*(WINDOW_AMP[0]+WINDOW_AMP[1]),
        mean,
        r'$A = %.1f$' % (2*damp) + "%",
        horizontalalignment='center',
        verticalalignment='center',
        bbox=props
    )
                                                                         # plot text mean value
    axlam.text(
        LAST_YEAR-2/main_freq,
        mean,
        r'$M = %.1f$' % (mean) + "%",
        horizontalalignment='left',
        verticalalignment='center',
        bbox=props,
    )
                                                                         # plot text in axlam1
    axlam1.set_ylim(axlam.get_ylim())
    ylims = axlam1.get_ylim()
    xlims = axlam1.get_xlim()
    mean_t = np.mean(relax_times)
    xcoord = time[0] + 0.25*(xlims[1]-xlims[0])
    yrange = ylims[1] - ylims[0]
    tmpb = label=="inf"
    for ii, st in enumerate(labels):
        axlam1.text(
            xcoord,
            ylims[ii] + (-1)**ii * 0.05 * yrange,
            r'$\alpha_{'+'{}'.format(st) + '}= %.1e$' % (ress[ii]["slope"]) + ' y$^{-1}$',
            horizontalalignment='center',
            verticalalignment='center',
            bbox=props,
        )
        axlam1.text(
            initial_times[ii] + relax_times[ii],
            mean + (-1)**ii * 0.01 * yrange,
            r'$\tau_{'+'{}'.format(st) + '}= %.1f$ y' % (relax_times[ii]),
            horizontalalignment='right' if ((tmpb and (ii==0)) or (not tmpb and (ii==1))) else 'left',
            verticalalignment='top' if ii==1 else 'bottom',
            bbox=props,
        )
                                                                         # plot (a) and (b)
    axlam.text(
        LAST_YEAR,
        ylims[-1],
        "(a)",
        horizontalalignment='left',
        verticalalignment='bottom',
    )
    axlam1.text(
        LAST_YEAR,
        ylims[-1],
        "(b)",
        horizontalalignment='left',
        verticalalignment='bottom',
    )
    gs.tight_layout(figlam, pad=0.0)

    return figlam, axlam, axlam1

def plot_IDEE(path, file_name, figure_name="omega.pdf", figure_name_all="all.pdf", show=False):
    """
    This function plots the main variables of IDEE.

    Input
        path : string
            the path of the data
        file_name : string
            name of file
        figure_name : string
            name of figure 'omega'
        figure_name_all : string
            name of figure 'all'
        show : boolean
            True if we show else save figures
    """
    badpath = road.join(path, "png_bad")
    goodpath = road.join(path, "png_good")
    namedir = road.join(path, RAW_PATH)
    if not road.exists(badpath):
        os.mkdir(badpath)
    if not road.exists(goodpath):
        os.mkdir(goodpath)
                                                                         # load data
    data = np.loadtxt(road.join(namedir, file_name), skiprows=1)
                                                                         # load variable names
    f = open(road.join(namedir, file_name), "r")
    lvars = f.readline()
    f.close()
    lvars = lvars.split('  ')[:-1]
    lvars = lvars[0].split(' ') + lvars[1:]
                                                                         # extend data
    lvars, data = extend_IDEE(lvars, data)
    time = data[:,0]
    select = (time >= (time[-1]+WINDOW_AMP[0])) * \
        (time <= (time[-1]+WINDOW_AMP[1]))
                                                                         # calculate the number of axes
    plot_list_keys = list(PLOT_LIST.keys())
    nb_p = len(plot_list_keys)
    nbrows = nb_p//NUMCOLS
    if nb_p%NUMCOLS > 0:
        nbrows += 1
                                                                         # make the figure
    fig, axes = plt.subplots(nbrows, NUMCOLS, sharex=True, figsize=(A4W, 4.))
    k = 0
    is_bad = False
                                                                         # check whether it is a bad attractor
    is_bad = check_bad_attractor(data)
    if is_bad:
        ylim_year = -1
        #ylim_year = 100*data[:, lvars.index("inflation")]
        #ylim_year = np.arange(time.size)[ylim_year<-15]
        #ylim_year = ylim_year[0]
    else:
        ylim_year = -1

    for i in range(nbrows):
        for j in range(NUMCOLS):
            ax = axes[i,j]
            try:
                var = plot_list_keys[k]
                if var=="debtratio":
                    raw = 1./data[:, lvars.index(var)]
                    mean, damp = comp_mean(raw, select)
                    ax.set_ylabel(r"$1/d$")
                elif var=="capital":
                    raw = data[:, lvars.index(var)]
                    mean = np.nan
                    ax.set_ylabel(r"$K$")
                else:
                    raw = 100*data[:, lvars.index(var)]
                    mean, damp = comp_mean(raw, select)
                    ax.set_ylabel(r"${}$".format(PLOT_LIST[var])) # var.replace("_", "\n"))

                mean = mean if not is_bad else np.nan
                raw[ylim_year:] = raw[ylim_year]
                ax.plot(time, raw, time, mean*np.ones(time.size))

            except IndexError:
                pass
            finally:
                k += 1
                                                                         # For omega,
                                                                         # compute and plot details in
                                                                         # another figure
    problem_detected = is_bad
    if not problem_detected:
        raw = 100*data[:, lvars.index("omega")]
        figlam, axlam, axlam1 = plot_main_details_IDEE(time, raw)
        problem_detected = figlam==0
    else:
        print("private-debt tipping points")
                                                                         # set the labels of the last row
    for j in range(NUMCOLS):
        ax = axes[nbrows-1, j]
        ax.set_xlabel("time")
    ax.set_xlim(time[0], LAST_YEAR)

    if not problem_detected:
        for axl in [axlam, axlam1]:
            axl.set_xlim(time[0], LAST_YEAR)
            axl.set_xlabel(r"$t$ (y)")
        axlam.set_ylabel(r"$\omega$ (%)")
        axlam1.legend(framealpha=1.)
        axlam.legend(framealpha=1., ncol=3)

        perso_savefig(figlam, goodpath, figure_name, show)
        name_path = goodpath
    else:
        name_path = badpath

    perso_savefig(fig, name_path, figure_name_all, show)

def plot_map(badc, goodc, path, figure_name="map.pdf", show=False):
    """
    This function is used to map the range of parameters.
    The criterion is good or bad attractors.

    Input
        badc : SALib.util.problem.ProblemSpec
            the class with all the bad attractor
        goodc : SALib.util.problem.ProblemSpec
            the class with all the good attractor
        path : string
            path of the data
        figure_name : string
            file name
        show : boolean
            if True, show else save
    Output
        isplot : boolean
            True if there is a plot, False else
    """
    if badc==None:
        print("There is no map to plot since there is no bad point.")
        return False

    names = badc["names"]
    bad_sample = badc.samples
    good_sample = goodc.samples
                                                                         # remove rows with nan
    select_rows = np.isnan(goodc.results).any(axis=1)
    nan_sample = good_sample[select_rows,:]
    good_sample = good_sample[~select_rows,:]
                                                                         # plot figure
    fig, ax = plt.subplots()

    ax.scatter(
        bad_sample[:, 0],
        bad_sample[:, 1],
        c='r',
        marker='X',
        label='private-debt tipping points'
    )
    ax.scatter(
        good_sample[:, 0],
        good_sample[:, 1],
        c='g',
        label='growth'
    )
    ax.scatter(nan_sample[:, 0], nan_sample[:, 1], c='b')

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.legend(framealpha=1.)

    perso_savefig(fig, path, figure_name, show) 

    return True

def plot_sa_class(sa_class, path, ylims=[YMIN, YMAX], figure_name="sa.pdf", show=False):
    """
    Plots sa_class.

    Input
        sa_class : SALib.util.problem.ProblemSpec
            the class with all the bad attractor
        path : string
            path of data
        figure_name : string
            file name
        show : boolean
            if True, show, else save fig
    """
    axes = sa_class.plot()
    axes[0,0].set_ylim(ylims[0], ylims[1])
    for row in axes:
        for col in row:
            xlims = col.get_xlim()
            col.plot([xlims[0], xlims[1]], [0., 0.])
            col.plot([xlims[0], xlims[1]], [1., 1.])

    #axes[0,0].set_yscale('log')
    plt.tight_layout(**PADS)

    fig = plt.gcf()
    perso_savefig(fig, path, figure_name, show)

def run_IDEE(sa_class, path):
    """
    This function makes a sensivity analysis of the model IDEE.

    Input
        sa_class : SALib.util.problem.ProblemSpec
            the class of the library SAlib
        path : string
            the  directory where to save IDEE's outputs
    """
                                                                         # define a global timer
    time_start = timer()
    nbmax = sa_class.samples.shape[0]
    name_dir = road.join(path, RAW_PATH)
    if not road.exists(name_dir):
        os.mkdir(name_dir)
                                                                         # run the model
    names = sa_class["names"]
    loop_times = []
    for n, sample in enumerate(sa_class.samples):
        loop_time_start = timer()
        print("\n--- {:d} / {:d} ---".format(n+1, nbmax))
        params = dict(zip(names, sample))
        IDEE(params, n, name_dir)
        loop_times.append(timer() - loop_time_start)

    mean_time_loop = np.mean(loop_times)
    print("\n  Mean execution of solving IDEE = {:.1e} s".format(mean_time_loop))
    print("  Total execution time = {:.1f} s".format(timer() - time_start))

def run_SA(sa_class, perc=[2, 98]):
    """
    Run the sensitivity analysis of the sa_class.

    Input
        sa_class : SALib.util.problem.ProblemSpec
            the class of the library SAlib
    Output
        sa_class : SALib.util.problem.ProblemSpec
            the class of the library SAlib
    """
                                                                         # copy class
    cc = ProblemSpec({
        "num_vars":sa_class["num_vars"],
        "names":sa_class["names"],
        "bounds":sa_class["bounds"],
        "outputs":sa_class["outputs"]
    })
                                                                         # get the parameters
    sorted_results = sa_class.results.copy()
    sorted_samples = sa_class.samples.copy()
    nbrows = sorted_results.shape[0]
                                                                         # remove extremal values
    for j in range(sorted_results.shape[1]):
        raw = sorted_results[:,j]
        lims = np.percentile(raw, perc)
        select = (raw>lims[0]) * (raw<lims[1])
        if select.any():
            sorted_results[~select,j] = np.nan
                                                                         # remove rows with nan
    select_rows = ~np.isnan(sorted_results).any(axis=1)
    print("  remove {:d} rows with nan".format(nbrows-np.count_nonzero(select_rows)))
    sorted_samples = sorted_samples[select_rows,:]
    sorted_results = sorted_results[select_rows,:]
                                                                         # make the good number of rows
    nb_rows = sorted_results.shape[0]
    Nrem = nb_rows%(2*cc["num_vars"]+2)
    print("  {:d} other samples to remove".format(Nrem))
    del_list = np.random.randint(low=0, high=nb_rows-1, size=Nrem)
    select_rows = [True]*nb_rows
    for i in del_list:
        select_rows[i] = False
    sorted_samples = sorted_samples[select_rows,:]
    sorted_results = sorted_results[select_rows,:]
                                                                         # add noise to the results (to remove divisions by zero)
    #noise = 1.E-15*np.random.randint(10, size=sorted_results.shape)
    #sorted_results += noise
                                                                         # update the class
    cc.set_samples(sorted_samples)
    cc.set_results(sorted_results)
    cc = comp_nb_samples(cc)
    cc["bad_array"] = np.zeros(cc["nb_samples"], dtype=np.bool8)
                                                                         # run the SA
    SA_time_start = timer()
    cc.analyze_sobol()
    SA_time = timer() - SA_time_start
    print("  Execution time of Sensitivity Analysis = {:.1e} s".format(SA_time))

    return cc

def save_sa_class(sa_class, path):
    """
    Save the class sa_class.

    Input
        sa_class : SALib.util.problem.ProblemSpec
            the class of the library SAlib
        path : string
            the path to save the class
    Output
        sa_class : SALib.util.problem.ProblemSpec
            the class of the library SAlib
    """
                                                                         # save the parameters names
    first_line = ""
    for var in sa_class["names"]:
        first_line += var + ' '
                                                                         # the sample
    np.savetxt(
        fname=road.join(path, "sample.dat"),
        X=sa_class.samples,
        header=first_line
    )
                                                                         # the bounds
    np.savetxt(
        fname=road.join(path, "bounds.dat"),
        X=np.asarray(sa_class["bounds"]),
    )
                                                                         # the results
    try:
        size = sa_class.results.size
        first_line = ""
        for out in sa_class["outputs"]:
            first_line += out + ' '

        np.savetxt(
            fname=road.join(path, "outputs_to_analyze.dat"),
            X=sa_class.results,
            header=first_line
        )
    except AttributeError:
        print("class is saved without outputs")
        pass

    try:
        bad_array = sa_class["bad_array"]
        np.savetxt(
            fname=road.join(path, "bad_array.dat"),
            X=bad_array,
        )
    except KeyError:
        print("class is saved without bad_array")
        pass

def set_results(sa_class, resdata, bad_array, outputs_name):
    """
    This function sets the array resdata as the results for SA.

    Input
        sa_class : SALib.util.problem.ProblemSpec
            the class of the library SAlib
        resdata : numpy.ndarray (float)
            the result for SA
        bad_array : numpy.ndarray (boolean)
            True if it is a simulation toward the bad attractor
        outputs_name ; list (string)
            outputs name
    Output
        sa_class : SALib.util.problem.ProblemSpec
            the class of the library SAlib
    """
    sa_class["outputs"] = outputs_name
    sa_class.set_results(resdata)
    sa_class["bad_array"] = bad_array.copy()

    return sa_class

def set_ylims(select, raw, ax):
    """
    Define the ylims on a given period.

    Input
        select : numpy.ndarray (boolean)
            the time window selection
        raw : numpy.ndarray (float)
            raw data
        ax : matplotlib.axes.Axes
            the axe
    """
    ax.set_ylim(
        bottom = np.nanmin(raw[select]),
        top=np.nanmax(raw[select])
    )

def sort_attractors(sa_class):
    """
    This function sorts the sa_class in two sets of parameters.
    It corresponds to two different attractors.

    Input
        sa_class : SALib.util.problem.ProblemSpec
            the class of the library SAlib
    Output
        bad_class : SALib.util.problem.ProblemSpec
            the class with all the bad attractor
        good_class : SALib.util.problem.ProblemSpec
            the class with all the good attractor
    """
    try:
        bad_array = sa_class["bad_array"]
    except KeyError:
        print("Cannot sort a class without its bad_array")
        return None, None
                                                                         # creates two classes
    bad_class = ProblemSpec({
        "num_vars":sa_class["num_vars"],
        "names":sa_class["names"],
        "bounds":sa_class["bounds"],
        "outputs":sa_class["outputs"]
    })
    good_class = ProblemSpec({
        "num_vars":sa_class["num_vars"],
        "names":sa_class["names"],
        "bounds":sa_class["bounds"],
        "outputs":sa_class["outputs"]
    })
                                                                         # sort results
    samples = sa_class.samples

    try:
        size = sa_class.results.size
        results = sa_class.results
    except AttributeError:
        print("Cannot find outputs, ignoring them.")
        results = np.zeros((samples.size, sa_class["num_vars"]), dtype=np.bool8)

    bad_r, good_r = [], []
    bad_s, good_s = [], []
    for n, samp in enumerate(samples):
        if bad_array[n]:
            bad_r.append(results[n,:])
            bad_s.append(samp)
        else:
            good_r.append(results[n,:])
            good_s.append(samp)

    bad_r = np.asarray(bad_r)
    bad_s = np.asarray(bad_s)
    good_r = np.asarray(good_r)
    good_s = np.asarray(good_s)
                                                                         # set new samples and results
    if bad_s.shape[0] > 0:
        bad_class.set_samples(bad_s)
        bad_class.set_results(bad_r)
        bad_class = comp_nb_samples(bad_class)
        bad_class["bad_array"] = np.ones(bad_class["nb_samples"], dtype=np.bool8)
    else:
        print("  there is no bad samples")
        bad_class = None

    if good_s.shape[0] > 0:
        good_class.set_samples(good_s)
        good_class.set_results(good_r)
        good_class = comp_nb_samples(good_class)
        good_class["bad_array"] = np.zeros(good_class["nb_samples"], dtype=np.bool8)
    else:
        print("  there is no good samples")
        good_class = None

    return bad_class, good_class

def test_IDEE():
    """
    This function does a simple test whether we can lunch IDEE.
    """
    name_dir = check_dir()
    for n, muvalues in enumerate([1.8, 2., 2.2]):
        params = {"mu":muvalues}
        args = {
            "params":params,
            "n":n,
            "name_dir":name_dir
        }
        IDEE(**args)
                                                                         # --- main -------------------------
if __name__=="__main__":
                                                                         # 1. Functions to make data for SA
    # 1.1 create a path
    path = check_dir()

    # 1.2 initialize a class
    sa_class = make_sa_class(POW)
    rep = input("\n  Number of samples is {:d}. Continue? Yes (Y) / No (N):\n".format(
        sa_class.samples.shape[0]))
    if not rep=="Y":
        sys.exit("You chose to quit.")

    # 1.3 save the class
    save_sa_class(sa_class, path)

    # 1.4 make a set of simulations
    run_IDEE(sa_class, path)

    # 1.5 compute the outputs
    resdata, bad_array, outputs_name = make_outputs(path)

    # 1.6 set the results in the class
    sa_class = set_results(sa_class, resdata, bad_array, outputs_name)

    # 1.7 save the class with the results
    save_sa_class(sa_class, path)
                                                                         # 2. Function to load data for SA
    sa_class_load = load_sa_class(path)
    print("  Is classes equal?", is_close(sa_class, sa_class_load))
                                                                         # 3. Perform the SA
    # 3.1 sort the good and bas attractors
    badc, goodc = sort_attractors(sa_class)

    # 3.2 make the sensitivity analysis on the good simulations
    sa_class = run_SA(goodc)
                                                                         # 4. Plots
    # 4.1 plot the map
    plot_map(badc, goodc, path)

    # 4.2 plot it
    plot_sa_class(sa_class, path)

    # 4.3 plot histograms
    plot_histo(sa_class, path)

    # 4.4 plot IDEE
    file_name = "gemmes.out.World_default_set0"
    plot_IDEE(path, file_name)
