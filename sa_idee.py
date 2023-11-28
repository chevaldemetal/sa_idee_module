"""
file name: sa_idee.py
language: Python 3
date: 20/11/2023
author: Hugo A. Martin
email: martin_hugo@ymail.com
description: This module is can be used to run the sensitivity analysis of the model IDEE.
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
                                                                         #--- macros for users --------------
POW = 5
NAMES = [
    "eta",
    "mu"
]
BOUNDS = [
    [0.16, 0.22],
    [1.5, 2.1]
]
OUTPUTS = [
"mean_lambda",
"mean_omega",
"main_frequency_of_cycles"
]
PLOT_LIST = [
    'capital',
    'g0',
    'omega',
    'debtratio',
    'lambda',
    'wage_growth',
    'productivity_growth',
    'smallpi',
    'kappa',
    'dividends_ratio',
    'inflation',
    'rb',
]
NUMCOLS = 3
LAST_YEAR = 3000
TMAX = 3000
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
IDEE = "./gemmes"
XMP_FILE = "gemmes.dat.example"
DAT_FILE = "gemmes.dat.World_default"
OUT_FILE = "gemmes.out.World_default"
DIR_IDEE = "/home/admin/Documents/sciences/postdocGU/codes/GEMMESCLIM/gemmes_cpl/sources/"
DIR_LOC = os.getcwd()
DIR_SAVE = road.join(DIR_LOC, "outputs_")
DT = 1./12
INFTY_SMALL = 1.E-12
EPSILON = 1.E-2

WINDOW_FREQ = 100
WINDOW_RELAX = 100
WINDOW_AMP = [400, 500]
WINDOW_MEAN = [400, 500]
A4W = 8.27
                                                                         #--- functions ---------------------
def f_check_bad_attractor(data, indices):
    """
    This function checks whether the simulation converges toward the bad attractor.

    ...

    Input
    -----
    data : numpy.ndarray (float)
        the array containing the simulation's data
    indices : list (float)
        the array of tricky variables' indices like lambda, capital etc.

    ...

    Output
    ------
    is_bad : boolean
        True wheter it converges toard the bad attractor
    """
    is_bad = False
    for ind in indices:
        raw = data[:, ind]
        is_bad = is_bad or (raw<INFTY_SMALL).any() or np.isnan(raw).any()

    return is_bad

def f_check_dir():
    """
    Check wether the outputs directory exists.

    ...

    Output
    ------
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

def f_comp_growth(data, nb):
    """
    Computes the growth rate of the array data.

    ...

    Input
    -----
    data : numpy.ndarray
        the raw data
    nb : integer
        the window length

    ...

    Output
    ------
    growth : numpy.ndarray
        the growth rate (same size filled with zeros)
    """
    g = (np.roll(data, -1) - data)/DT/data
    growth = np.convolve(g[::-1], np.ones(nb), "valid") / nb
    growth[0] = np.nan
    growth = np.hstack(([np.nan]*(nb-1), growth))[::-1]

    return growth

def f_comp_main_freq(signal):
    """
    Computes the main frequency of the signal.

    ...

    Input
    -----
    signal : numpy.ndarray
        the signal

    ...

    Output
    ------
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

def f_comp_observations(time, raw):
    """
    Compute the observations on the lambda variable.

    ...

    Input
    -----
    time : numpy.ndarray (float)
        time
    raw : numpy.ndarray (float)
        the raw signal (most of the time 'lambda')

    ...

    Output
    ------
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
        converge : boolean
            true whether the signal converges toward the mean value
        initial_times : list (float)
            the times from which it converges
    """
    try:
                                                                         # selects
        selectm = (time >= (time[0]+WINDOW_MEAN[0])) * \
            (time <= (time[0]+WINDOW_MEAN[1]))
        selectf = time <= (time[0] + WINDOW_FREQ)
        selectamp = (time >= (time[0]+WINDOW_AMP[0])) * \
            (time <= (time[0]+WINDOW_AMP[1]))
                                                                         # compute the mean
        mean = np.mean(raw[selectm])
        ismean = (np.abs(raw - mean)<0.01)
                                                                         # compute the main frequency
        main_freq, sp, f = f_comp_main_freq(raw[selectf])
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
        anal_sup = hilbert(raw-mean)
        big_env_sup = np.abs(anal_sup)
        tmp_sel = time <= (time[0]+100)
        #amp_ini = mean_sup_ini
        nanmax_ini, nanmin_ini = np.nanmax(raw[tmp_sel]), np.nanmin(raw[tmp_sel])
        amp_ini = nanmax_ini - nanmin_ini

        tmp_sel = ((time[0]+300) <= time) * (time <= (time[0]+400))
        mean_sup_infty = np.nanmean(big_env_sup[tmp_sel])
        #amp_infty = mean_sup_infty
        nanmax_infty, nanmin_infty = np.nanmax(raw[tmp_sel]), np.nanmin(raw[tmp_sel])
        amp_infty = nanmax_infty - nanmin_infty

        converge = amp_ini/amp_infty > 1.05

        if False:
            print(converge)

            fig, ax = plt.subplots()
            ax.plot(time, raw-mean)
            ax.plot(time, big_env_sup)
            #ax.plot(time, big_env_inf)

            ax.plot([time[0], time[0]+100], [nanmax_ini]*2)
            ax.plot([time[0], time[0]+100], [nanmin_ini]*2)
            ax.plot([time[0]+300, time[0]+400], [nanmax_infty]*2)
            ax.plot([time[0]+300, time[0]+400], [nanmin_infty]*2)

            plt.show()
            sys.exit()

        selpos = ((raw - mean) >= 0.)
        selneg = ((raw - mean) <= 0.)
        changing_signs = (np.roll(gradient, -1) * gradient)<=0.
        sel_sup = changing_signs * selpos
        sel_inf = changing_signs * selneg

        tinf = time[sel_inf]
        rawinf = raw[sel_inf]
        if converge:
            sel_inf_2 = np.logical_or(
                ((np.roll(rawinf, -1) - rawinf)>=0.),
                np.abs(rawinf-mean)<0.1
            )
            sel_inf_2[0] = rawinf[0] <= rawinf[1]
        else:
            sel_inf_2 = [True]*tinf.size
        rawinf = rawinf[sel_inf_2]
        tinf = tinf[sel_inf_2]

        tsup = time[sel_sup]
        rawsup = raw[sel_sup]
        if converge:
            sel_sup_2 = np.logical_or(
                ((np.roll(rawsup, -1) - rawsup)<=0.),
                np.abs(rawsup-mean)<0.1
            )
            sel_sup_2[0] = rawinf[0] >= rawinf[1]
        else:
            sel_sup_2 = [True]*tsup.size
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
            if ii==0:
                if converge:
                    y = nanmax
                    scale = 1.
                else:
                    y = nanmin
                    scale = -1.
            else:
                if converge:
                    y = nanmin
                    scale = -1.
                else:
                    y = nanmax
                    scale = 1.

            Delta = np.gradient(amp_env, DT)
            Delta[np.logical_not(np.sign(Delta)==scale)] = 0.

            argD = np.nanargmax(np.abs(Delta))
            slope = Delta[argD]
            not_zero = np.abs(Delta)>INFTY_SMALL

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
                                                                         # compute the main amplitude
        damp = 0.5*(np.mean(xs[selectamp]) - np.mean(xi[selectamp]))

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
            "converge":converge,
            "initial_times":initial_times
        }

    except Exception:
        print("Problem in 'f_comp_observations'")
        if RAISE:
            print(outs)
            print(buffs)

            raise

    return outs, buffs

def f_empty(a=0,b=0,c=0,d=0,e=0,f=0):
    pass

def f_extend_IDEE(lvars, data):
    """
    This function adds new fields compputed from the raw data.

    ...

    Input
    -----
    lvars : list (string)
        the list of variables names
    data : numpy.ndarray (float)
        the raw data

    ...

    Output
    ------
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
    new[:,0] = f_comp_growth(data[:,4], nb)
    data = np.hstack((data, new))
                                                                         # append productivity growth
    lvars.append("productivity_growth")
    new = np.zeros((nbrows,1))
    new[:,0] = f_comp_growth(data[:,5], nb)
    data = np.hstack((data, new))
                                                                         # append population growth
    lvars.append("population_growth")
    new = np.zeros((nbrows,1))
    new[:,0] = f_comp_growth(data[:,2], nb)
    data = np.hstack((data, new))
    return lvars, data

def f_IDEE(params, n, name_dir):
    """
    This function defines a set of params, sets a single problem to simulate and runs it.

    ...

    Argument
    --------
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
    subprocess.run(IDEE)
                                                                         # save the outputs
    shutil.copyfile(OUT_FILE, road.join(name_dir, OUT_FILE + '_set{:d}'.format(n)))

    os.chdir(DIR_LOC)

def f_is_close(c1, c2):
    """
    Returns True whether the classes sa_c1 and sa_c2 are similar.

    ...

    Input
    -----
    c1 : SALib.util.problem.ProblemSpec
        first class of the library SAlib
    c2 : SALib.util.problem.ProblemSpec
        second class of the library SAlib

    ...

    Output
    ------
    is_close : boolean
        True whether the two classes are equal (with a controled error)
    """
    error = lambda b, k: print(" {} are different".format(k)) if (not b) else 0
                                                                         # list of cheks
                                                                         # num_vars
    k = "num_vars"
    error(c1[k]==c2[k], k)
                                                                         # names and outputs
    for k in ["names", "outputs"]:
        n1 = c1[k]
        n2 = c2[k]
        for n, name in enumerate(n1):
            error(name==n2[n], k)
                                                                         # bounds
    k = "bounds"
    error(np.allclose(c1[k], c2[k]), k)
                                                                         # results
    try:
        s1 = c1.results.size
        s2 = c2.results.size
        k = "results array"
        error(s1==s2, k)
        error(np.allclose(c1.results, c2.results), k)

    except AttributeError:
        print("At least one class does not have any results array")

    return True

def f_make_outputs(path, plot_freq=False):
    """
    This function loads a sample of simulations and make the result array.

    ...

    Input
    -----
    path : string
        location of the simulations
    plot_freq : boolean
        if True, plot the approximated signal with the main frequency then quit

    ...

    Output
    ------
    result : numpy.ndarray (float)
        the array of resulted post-treatment data
    """
    func = f_plot_main_freq if plot_freq else f_empty

    samples = np.loadtxt(road.join(path, "sample.dat"))
    nums = samples.shape[0]

    mean_l = []
    mean_o = []
    cycle_freq = []
    print("  {:d} simulations to do".format(nums))
    for n in range(nums):
        if n%10==0:
            print('   {:d}'.format(n))

        data = np.loadtxt(road.join(path,
            OUT_FILE+'_set{:d}'.format(n)),
            skiprows=1
        )
        sample = [n] + samples[n,:].tolist()

        time = data[:,0]
        omega = data[:,18]
        lambd = data[:,19]
                                                                         # get means
        select = (time >= (time[0]+WINDOW_MEAN[0])) * \
            (time <= (time[0]+WINDOW_MEAN[1]))
        m_l = np.mean(lambd[select])
        m_o = np.mean(omega[select])
                                                                         # get the main frequency
        select = time <= (time[0] + WINDOW_FREQ)
        main_freq, sp, freq = f_comp_main_freq(omega[select])
                                                                         # plot of required
        func(main_freq, sp, freq, omega, time, sample)

        mean_l.append(m_l)
        mean_o.append(m_o)
        cycle_freq.append(main_freq)
                                                                         # turn lists in arrays
    mean_l = np.asarray(mean_l)
    mean_o = np.asarray(mean_o)
    cycle_freq = np.asarray(cycle_freq)
    results = np.column_stack((mean_l, mean_o, cycle_freq))

    return results

def f_make_sa_class(name_dir, Pow=2):
    """
    This function makes a class of the library SAlib with the required parameters.

    ...

    Input
    -----
    Pow : integer
        the power used to determine the sample size

    ...

    Ouput
    -----
    sa_class : SALib.util.problem.ProblemSpec
        the class of the library SAlib
    """
    sa_class = ProblemSpec({
        "num_vars":2,
        "names":NAMES,
        "bounds":BOUNDS,
        "outputs":OUTPUTS,
    })
                                                                         # get the parameters
    sa_class.sample_saltelli(N=2**Pow)
    print("  sa_class made with {:d} samples".format(
        sa_class.samples.shape[0]
    ))

    return sa_class

def f_load_sa_class(path):
    """
    This functions loads the data and make a sa_class.

    ...

    Input
    -----
    path : string
        the path where to load the data

    ...

    Output
    ------
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

    filename = road.join(path, "outputs_to_analyze.dat")
    if road.exists(filename):
        results = np.loadtxt(filename)
        f = open(filename)
        out_names = f.readline().split(' ')[1:-1]
        f.close()
        sa_class["outputs"] = out_names
        sa_class.set_results(results)

    return sa_class

def f_plot_main_freq(main_freq, sp, freq, signal, time, sample):
    """
    Plot the approximated signal thanks to the main frequency.

    ...

    Arguments
    ---------
    main_freq : float
        the main frequency computed by the Fourier transform
    sp : numpy.ndarray (float)
        Fourier coefficients
    freq : numpy.ndarray (float)
        Frequencies
    signal : numpy.ndarray (float)
        the original signal
    time : numpy.ndarray (float)
        the time
    sample : numpy.ndarray
        the array of parameters, first element is the sample number
    """

    fig, axes = plt.subplots(nrows=2, ncols=1)

    axes[0].plot(freq, np.absolute(sp)[1:])
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("absolute Fourier coefficient value")

    mean_s = np.mean(signal)
    analytic_signal = hilbert(signal - mean_s)
    amplitude_envelope = mean_s + np.abs(analytic_signal)

    amp = 0.5*np.max(np.abs(signal) - mean_s)
    approx_s = mean_s + amp*np.sin(2.*np.pi*main_freq*time)
    Tmax = time[np.argmax(signal/amplitude_envelope/time)]
    delay = Tmax - time[0]

    axes[1].plot(time, signal, label='original signal')
    #axes[1].plot(time, amplitude_envelope, label='signal envelope')
    axes[1].plot(time+delay, approx_s, label='$C^0 \sin(2\pi f_m t)$')
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("main frequency signal")
    axes[1].legend()

    print(sample[0], sample[1:])
    plt.show()
    plt.close(fig)
    rep = input("  Press Enter to continue, Q to quit: ")
    if rep=="Q":
        sys.exit("  Execution stopped by User")

def f_plot_main_details_IDEE(time, raw):
    """
    Return a figure that shows the main detail of the computed quantities
    for the SA of IDEE.

    ...

    Input
    -----
    time : numpy.ndarray (float)
        time
    raw : numpy.ndarray (float)
        the raw signal (most of the time 'lambda')

    ...

    Output
    ------
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
    outs, buffs = f_comp_observations(time, raw)

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
    converge = buffs["converge"]
    initial_times = buffs["initial_times"]

    t0 = initial_times[0] if label=="inf" else initial_times[1]
    labels = ["inf", "sup"]
    selectr = time <= (time[0] + WINDOW_RELAX)
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
    tmp = ress[0]["intercept"] + ress[0]["slope"]*time[st]
    nanminxi = np.nanmin(xi)
    nanmaxxi = np.nanmax(xi)
    if converge:
        tmpsel = tmp <= nanmaxxi
        toplot = nanmaxxi
    else:
        tmpsel = tmp >= nanminxi
        toplot = nanminxi
    lineres, = axlam1.plot(timest[tmpsel], tmp[tmpsel],
        color="C2"
    )
    axlam1.plot(timest, toplot*np.ones(sst),
        color=lineres.get_color(),
        linestyle=lineres.get_linestyle(),
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
    if converge:
        tmpsel = tmp >= nanminxs
        toplot = nanminxs
    else:
        tmpsel = tmp <= nanmaxxs
        toplot = nanmaxxs
    axlam1.plot(timest[tmpsel], tmp[tmpsel],
        color=lineres.get_color(),
        linestyle=lineres.get_linestyle(),
    )
    axlam1.plot(timest, toplot*np.ones(sst),
        color=lineres.get_color(),
        linestyle=lineres.get_linestyle(),
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
        [time[0]+WINDOW_AMP[0], time[0]+WINDOW_AMP[1]],
        [mean+damp, mean+damp],
    )
    axlam.plot(
        [time[0]+WINDOW_AMP[0], time[0]+WINDOW_AMP[1]],
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
        [time[0]+0.5*(WINDOW_AMP[0]+WINDOW_AMP[1])]*2,
        [mean+damp, mean],
        color=line.get_color(),
        linestyle=line.get_linestyle()
    )
    axlam.plot(
        [time[0]+0.5*(WINDOW_AMP[0]+WINDOW_AMP[1])]*2,
        [mean, mean-damp],
        color=line.get_color(),
        linestyle=line.get_linestyle()
    )
    axlam.text(
        time[0]+0.5*(WINDOW_AMP[0]+WINDOW_AMP[1]),
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

def f_plot_IDEE(path, file_name, figure_name, figure_name_all, savefig=False):
    """
    This function plots the main variables of IDEE.

    ...

    Input
    -----
    path : string
        the path of the data
    file_name : string
        name of file
    figure_name : string
        name of figure 'lambda'
    figure_name_all : string
        name of figure 'all'
    savefig : boolean
        True if we save the figure, show if False
    """
    badpath = road.join(path, "png_bad")
    goodpath = road.join(path, "png_good")
    if not road.exists(badpath):
        os.mkdir(badpath)
    if not road.exists(goodpath):
        os.mkdir(goodpath)
                                                                         # load data
    data = np.loadtxt(road.join(path, file_name), skiprows=1)
                                                                         # load variable names
    f = open(road.join(path, file_name), "r")
    lvars = f.readline()
    f.close()
    lvars = lvars.split('  ')[:-1]
    lvars = lvars[0].split(' ') + lvars[1:]
                                                                         # extend data
    lvars, data = f_extend_IDEE(lvars, data)
    time = data[:,0]
    select = (time >= (time[0]+WINDOW_MEAN[0])) * \
        (time <= (time[0]+WINDOW_MEAN[1]))
                                                                         # calculate the number of axes
    nb_p = len(PLOT_LIST)
    nbrows = nb_p//NUMCOLS
    if nb_p%NUMCOLS > 0:
        nbrows += 1
                                                                         # make the figure
    fig, axes = plt.subplots(nbrows, NUMCOLS, sharex=True, figsize=(1.2*A4W, 4))
    k = 0
    is_bad = False
    tricky_vars = ["capital", "omega", "lambda"]
    indices = []
    for var in tricky_vars:
        indices.append(lvars.index(var))
    is_bad = f_check_bad_attractor(data, indices)

    for i in range(nbrows):
        for j in range(NUMCOLS):
            ax = axes[i,j]
            try:
                var = PLOT_LIST[k]
                if not var=="capital":
                    raw = 100*data[:, lvars.index(var)]
                else:
                    raw = data[:, lvars.index(var)]

                mean = np.mean(raw[select])
                ax.plot(time, raw, time, mean*np.ones(time.size))
                ax.set_ylabel(var)
            except IndexError:
                pass
            finally:
                k += 1
                                                                         # For lambda,
                                                                         # compute and plot details in
                                                                         # another figure
    problem_detected = is_bad
    if not problem_detected:
        raw = 100*data[:, lvars.index("lambda")]
        figlam, axlam, axlam1 = f_plot_main_details_IDEE(time, raw)
        problem_detected = figlam==0
    else:
        print("financial tiping point")
                                                                         # set the labels of the last row
    for j in range(NUMCOLS):
        ax = axes[nbrows-1, j]
        ax.set_xlabel("time")
    ax.set_xlim(time[0], LAST_YEAR)

    if not problem_detected:
        for axl in [axlam, axlam1]:
            axl.set_xlim(time[0], LAST_YEAR)
            axl.set_xlabel(r"$t$ (y)")
        axlam.set_ylabel(r"$\lambda$ (%)")
        axlam1.legend(framealpha=1.)
        axlam.legend(framealpha=1., ncol=3)

    if savefig:

        if not problem_detected:
            name = road.join(goodpath, figure_name)
            print("saving of {}".format(figure_name))
            plt.savefig(name,
                bbox_inches="tight", pad_inches=0.1,
            )
            plt.close(figlam)
            name = road.join(goodpath, figure_name_all)
        else:
            name = road.join(badpath, figure_name_all)

        print("saving of {}".format(figure_name_all))
        plt.savefig(name,
            bbox_inches="tight", pad_inches=0.1,
        )
        plt.close(fig)
    else:
                                                                         # show
        ax.set_xlim(time[0], LAST_YEAR)
        plt.show()
        plt.close('all')

def f_plot_map(badc, goodc):
    """
    This function is used to map the range of parameters.
    The criterion is good or bad attractors.

    ...

    Input
    -----
    badc : SALib.util.problem.ProblemSpec
        the class with all the bad attractor
    goodc : SALib.util.problem.ProblemSpec
        the class with all the good attractor
    """
    names = badc["names"]
    bad_sample = badc.samples
    good_sample = goodc.samples
                                                                         # remove rows with nan
    select_rows = np.isnan(goodc.results).any(axis=1)
    nan_sample = good_sample[select_rows,:]
    good_sample = good_sample[~select_rows,:]
                                                                         # plot figure
    fig, ax = plt.subplots()
    coords = np.vstack((bad_sample, good_sample, nan_sample))
    colors = ['b']*bad_sample.shape[0] + ['g']*good_sample.shape[0] + ['r'] * nan_sample.shape[0]
    s = np.ones(coords.shape[0])
    ax.scatter(coords[:,0], coords[:,1], c=colors)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    plt.show()
    plt.close(fig)

def f_run_IDEE(sa_class, name_dir):
    """
    This function makes a sensivity analysis of the model IDEE.

    ...

    Input
    -----
    sa_class : SALib.util.problem.ProblemSpec
        the class of the library SAlib
    name_dir : string
        the  directory where to save IDEE's outputs
    """
                                                                         # define a global timer
    time_start = timer()
                                                                         # run the model
    names = sa_class["names"]
    loop_times = []
    for n, sample in enumerate(sa_class.samples):
        loop_time_start = timer()

        params = dict(zip(names, sample))
        f_IDEE(params, n, name_dir)
        loop_times.append(timer() - loop_time_start)

    mean_time_loop = np.mean(loop_times)
    print("\n  Mean execution of solving IDEE = {:.1e} s".format(mean_time_loop))
    print("  Total execution time = {:.1f} s".format(timer() - time_start))

def f_run_SA(sa_class):
    """
    Run the sensitivity analysis of the sa_class.

    ...

    Input
    -----
    sa_class : SALib.util.problem.ProblemSpec
        the class of the library SAlib

    ...

    Output
    ------
    sa_class : SALib.util.problem.ProblemSpec
        the class of the library SAlib
    """
    sorted_results = sa_class.results.copy()
    sorted_samples = sa_class.samples.copy()
                                                                         # remove rows with nan
    select_rows = ~np.isnan(sa_class.results).any(axis=1)
    sorted_samples = sorted_samples[select_rows,:]
    sorted_results = sorted_results[select_rows,:]
                                                                         # make the good number of rows
    nb_rows = sorted_results.shape[0]
    Nrem = nb_rows%(2*sa_class["num_vars"]+2)
    print("  {:d} samples to remove".format(Nrem))
    del_list = np.random.randint(low=0, high=nb_rows-1, size=Nrem)
    select_rows = [True]*nb_rows
    for i in del_list:
        select_rows[i] = False
    sorted_samples = sorted_samples[select_rows,:]
    sorted_results = sorted_results[select_rows,:]
                                                                         # update the class
    sa_class.set_samples(sorted_samples)
    sa_class.set_results(sorted_results)
                                                                         # run the SA
    SA_time_start = timer()
    sa_class.analyze_sobol()
    SA_time = timer() - SA_time_start
    print("  Execution time of Sensitivity Analysis = {:.1e} s".format(SA_time))

    return sa_class

def f_save_sa_class(sa_class, path):
    """
    Save the class sa_class.

     ...

    Input
    -----
    sa_class : SALib.util.problem.ProblemSpec
        the class of the library SAlib
    path : string
        the path to save the class

    ...

    Output
    ------
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
        pass

    return sa_class

def f_set_results(sa_class, resdata):
    """
    This function sets the array resdata as the results for SA.

    ...

    Input
    -----
    sa_class : SALib.util.problem.ProblemSpec
        the class of the library SAlib
    resdata : numpy.ndarray (float)
        the result for SA

    ...

    Output
    ------
    sa_class : SALib.util.problem.ProblemSpec
        the class of the library SAlib
    """
    sa_class.set_results(resdata)

    return sa_class

def f_sort_attractors(sa_class):
    """
    This function sorts the sa_class in two sets of parameters.
    It corresponds to two different attractors.

    ...

    Input
    -----
    sa_class : SALib.util.problem.ProblemSpec
        the class of the library SAlib

    ...

    Output
    ------
    bad_class : SALib.util.problem.ProblemSpec
        the class with all the bad attractor
    good_class : SALib.util.problem.ProblemSpec
        the class with all the good attractor
    """
    l_outs = sa_class["outputs"]
    if not ("mean_lambda" in l_outs):
        raise ValueError("mean_lambda must be in SAlib class outputs")
    else:
        ind = l_outs.index("mean_lambda")
                                                                         # creates two classes
    bad_class = ProblemSpec({
        "num_vars":len(sa_class["names"]),
        "names":sa_class["names"],
        "bounds":sa_class["bounds"],
        "outputs":sa_class["outputs"]
    })
    good_class = ProblemSpec({
        "num_vars":len(sa_class["names"]),
        "names":sa_class["names"],
        "bounds":sa_class["bounds"],
        "outputs":sa_class["outputs"]
    })
                                                                         # sort results
    results = sa_class.results
    sample = sa_class.samples

    data = sa_class.results[:,ind]
    bad_r, good_r = [], []
    bad_s, good_s = [], []
    for n, res in enumerate(data):
        is_bad = abs(res)<INFTY_SMALL or np.isnan(res).any()
        if is_bad:
            bad_r.append(results[n,:])
            bad_s.append(sample[n,:])
        else:
            good_r.append(results[n,:])
            good_s.append(sample[n,:])

    bad_r = np.asarray(bad_r)
    bad_s = np.asarray(bad_s)
    good_r = np.asarray(good_r)
    good_s = np.asarray(good_s)
                                                                         # set new samples and results
    if bad_s.shape[0] > 0:
        bad_class.set_samples(bad_s)
        bad_class.set_results(bad_r)
    else:
        print("  there is no bad samples")

    if good_s.shape[0] > 0:
        good_class.set_samples(good_s)
        good_class.set_results(good_r)
    else:
        print("  there is no good samples")

    return bad_class, good_class

def f_test_f_IDEE():
    """
    This function does a simple test whether we can lunch IDEE.
    """
    name_dir = f_check_dir()
    for n, muvalues in enumerate([1.8, 2., 2.2]):
        params = {"mu":muvalues}
        args = {
            "params":params,
            "n":n,
            "name_dir":name_dir
        }
        f_IDEE(**args)
                                                                         # --- main -------------------------
if __name__=="__main__":
                                                                         # 1 Functions to make data for SA
    path = f_check_dir()
    sa_class = f_make_sa_class(path, POW)
    f_save_sa_class(sa_class, path)
    f_run_IDEE(sa_class, path)
    resdata = f_make_outputs(path, plot_freq=False)
    sa_class = f_set_results(sa_class, resdata)
    f_save_sa_class(sa_class, path)
                                                                         # 2 functions to load data for SA
    #sa_class_load = f_load_sa_class(path)
    #print("  Is classes equal?", f_is_close(sa_class, sa_class_load))
                                                                         # 3 perform the SA
    #sa_class = f_run_SA(sa_class)
    badc, goodc = f_sort_attractors(sa_class)
    f_plot_map(badc, goodc)
