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
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib import ProblemSpec
from timeit import default_timer as timer
from scipy.signal import hilbert
import os.path as road
                                                                         #--- macros ------------------------
IDEE = "./gemmes"
XMP_FILE = "gemmes.dat.example"
DAT_FILE = "gemmes.dat.World_default"
OUT_FILE = "gemmes.out.World_default"
DIR_IDEE = "/home/admin/Documents/sciences/postdocGU/codes/GEMMESCLIM/gemmes_cpl/sources/"
DIR_LOC = os.getcwd()
DIR_SAVE = road.join(DIR_LOC, "outputs_")
DT = 1./12
INFTY_SMALL = 1.E-12

NAMES = ["eta", "mu"]
BOUNDS = [[0.16, 1.],
          [1.5, 2.5]]
OUTPUTS = ["mean_lambda", "mean_omega", "main_frequency_of_cycles"]
                                                                         #--- functions ---------------------
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

def f_empty(a=0,b=0,c=0,d=0,e=0,f=0):
    pass

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

    sample = np.loadtxt(road.join(path, "sample.dat"))
    nums = sample.shape[0]

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

        time = data[:,0]
        omega = data[:,18]
        lambd = data[:,19]
                                                                         # get means
        select = time>2800
        m_l = np.mean(lambd[select])
        m_o = np.mean(omega[select])
                                                                         # get the main frequency
        select = time<2200
        sp = np.fft.rfft(omega[select])
        nf = np.argmax(np.absolute(sp)[1:])
        freq = np.fft.rfftfreq(omega[select].size, d=DT)[1:]
        main_freq = freq[nf]
                                                                         # plot of required
        func(main_freq, sp, freq, omega, time)

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

def f_plot_main_freq(main_freq, sp, freq, signal, time):
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
    """

    fig, axes = plt.subplots(nrows=2, ncols=1)

    axes[0].plot(freq, np.absolute(sp)[1:])
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("absolute Fourier coefficient value")

    mean_s = np.mean(signal)
    analytic_signal = hilbert(signal - mean_s)
    amplitude_envelope = mean_s + np.abs(analytic_signal)

    amp = np.max(np.abs(signal) - mean_s)
    approx_s = mean_s + amp*np.sin(2.*np.pi*main_freq*time)
    Tmax = time[np.argmax(signal/amplitude_envelope/time)]
    delay = Tmax - (time[0] + 1./4/main_freq)

    axes[1].plot(time, signal, label='original signal')
    #axes[1].plot(time, amplitude_envelope, label='signal envelope')
    axes[1].plot(time+delay, approx_s, label='$C^0 \sin(2\pi f_m t)$')
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("main frequency signal")
    axes[1].legend()

    plt.show()
    plt.close(fig)
    rep = input("Press Enter to continue, Q to quit: ")
    if rep=="Q":
        sys.exit("Execution stopped by User")

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
    data = sa_class.results[:,ind]
    sample = sa_class.samples

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
    sa_class = f_make_sa_class(path, 7)
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
