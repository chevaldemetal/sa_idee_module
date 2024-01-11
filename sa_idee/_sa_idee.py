"""
file name: sa_idee.py
language: Python 3
date: 20/11/2023
author: Hugo A. Martin
email: martin_hugo@ymail.com
description: This module is can be used to run the sensitivity analysis of the model IDEE.

TODO:

-- paraléliser le programme
"""
                                                                         #--- imports -----------------------
from sa_idee._libs import *
from sa_idee._plot_func import *
from sa_idee._utils import *
                                                                         #--- macros for users --------------
                                                                         #--- functions ---------------------
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

def comp_nb_groups(c):
    """
    Compute the number of groups a sa_class has.

    Input
        c : salib.util.problem.problemspec
            class of the library salib
    Ouput
        c : salib.util.problem.problemspec
            class updated with nb_samples
    """
    n = len(list(dict.fromkeys(c["groups"])))
    c["nb_groups"] = n

    return c

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

def empty_print(a):
    print(a)
    return 1

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
                                                                         # create a working directory
    os.chdir(DIR_IDEE)
    wdir = "simu_{:d}".format(n)
    os.mkdir(wdir)
    shutil.copyfile(XMP_FILE, road.join(wdir, DAT_FILE))
    shutil.copyfile(REG_FILE, road.join(wdir, REG_FILE))
    shutil.copy(COM_IDEE, road.join(wdir, COM_IDEE))
    os.chdir(wdir)
                                                                         # change the file of parameters
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
                                                                         # delete the working dir
    shutil.rmtree(road.join(DIR_IDEE, wdir))

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
    for k in ["names", "outputs", "groups"]:
        n1 = c1[k]
        n2 = c2[k]
        for n, name in enumerate(n1):
            is_close += error(name==n2[n], k)
                                                                         # bounds
    k = "bounds"
    is_close += error(np.allclose(c1[k], c2[k]), k)
                                                                         # samples
    k = "samples"
    is_close += error(np.allclose(c1.samples, c2.samples), k)
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
    time_start = timer()

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
    f = open(road.join(name_dir, OUT_FILE+'_set0'), "r")
    lvars = f.readline()
    f.close()
    lvars = lvars.split('  ')[:-1]
    lvars = lvars[0].split(' ') + lvars[1:]
    for n in range(first_ind, nums):
                                                                         # load here
        file_name = road.join(name_dir, OUT_FILE+'_set{:d}'.format(n))
        data = np.loadtxt(file_name, skiprows=1)
                                                                         # check whether it is bad attracted
        bad_array[n] = check_bad_attractor(data, lvars)
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

    print("  Total execution time for outputs = {:.1f} s".format(timer() - time_start))

    return results, bad_array, keys

def make_outputs_multiproc(path, nb_cpu=4):
    """
    Parallel version of the make_outputs_multiproc.

    Input
        path : string
            location of the simulations
        nb_cpu : integer
            number of CPUs to be used
    Output
        result : numpy.ndarray (float)
            the array of resulted post-treatment data
        bad_array : numpy.ndarray (boolean)
            True if it is a simulation toward the bad attractor
        keys : list (string)
            outputs names
    """
    time_start = timer()
    
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

    print("  {:d} simulations to post process".format(nums-first_ind))
    f = open(road.join(name_dir, OUT_FILE+'_set0'), "r")
    lvars = f.readline()
    f.close()
    lvars = lvars.split('  ')[:-1]
    lvars = lvars[0].split(' ') + lvars[1:]
                                                                         # the parallel loop is here

    args = [
        [
            road.join(name_dir, OUT_FILE+'_set{:d}'.format(n)),
            lvars,
            output_list,
            n
        ]
        for n in range(first_ind, nums)
    ]


    with Pool(nb_cpu) as pool:
        for n, is_bad, outputs in pool.imap(make_outputs_multiproc_f, args):

            bad_array[n] = is_bad

            line = []
            for key in keys:
                line.append(outputs[key])
                                                                         # results are stored here
            results[n, :] = line
                                                                         # save the temporary file all 10th
            if (n%10==0) or (n==nums-1):
                print('   {:d}'.format(n))
                np.savetxt(fname=tmp_file_name, X=results)
                np.savetxt(fname=tmp_bad_name, X=bad_array)

    print("  Total execution time for outputs = {:.1f} s".format(timer() - time_start))

    return results, bad_array, keys

def make_outputs_multiproc_f(arg):
    """
    The function inside the parallel loop.

    Input
        arg : list (*)
            [file_name, ]
    Ouput
        n : integer
            the number of sample
        is_bad : boolean
            True if it is a bad attractor
        outputs : dict (*)
            the output of observations
    """
    file_name = arg[0]
    lvars = arg[1]
    output_list = arg[2]
    n = arg[3]
                                                                         # load here
    data = np.loadtxt(file_name, skiprows=1)
                                                                         # check whether it is bad attracted
    is_bad = check_bad_attractor(data, lvars)
                                                                         # extend data
    lvars, data = extend_IDEE(lvars, data)
    time = data[:,0]
    selectamp = (time >= (time[-1]+WINDOW_AMP[0])) * (time <= (time[-1]+WINDOW_AMP[1]))

    outputs = {}
    if not is_bad:
        for var in output_list:
            raw = 100.*data[:, lvars.index(var)]
                                                                         # computations are here
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

    return n, is_bad, outputs

def make_sa_class(Pow=2, names=NAMES, groups=GROUPS, bounds=BOUNDS):
    """
    This function makes a class of the library SAlib with the required parameters.

    Input
        Pow : integer
            the power used to determine the sample size
        names : list (string)
            name of variables in the SA
        groups : list (string)
            names of groups
        bounds : list (integer)
            bounds of the variables
    Ouput
        sa_class : SALib.util.problem.ProblemSpec
            the class of the library SAlib
    """
    sa_class = ProblemSpec({
        "num_vars":len(names),
        "names":names,
        "groups":groups,
        "bounds":bounds,
        "outputs":OUTPUTS,
    })
                                                                         # get the parameters
    sa_class.sample_saltelli(N=2**Pow)
    print("  sa_class made with {:d} samples".format(
        sa_class.samples.shape[0]
    ))
    sa_class = comp_nb_samples(sa_class)
    sa_class = comp_nb_groups(sa_class)

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
    groups = f.readline().split(' ')[1:-1]
    f.close()

    filename = road.join(path, "bounds.dat")
    bounds = np.loadtxt(filename)

    sa_class = ProblemSpec({
        "num_vars":len(names),
        "names":names,
        "groups":groups,
        "bounds":bounds.tolist(),
        "outputs":["None"]
    })
    sa_class.set_samples(sample)
    sa_class = comp_nb_samples(sa_class)
    sa_class = comp_nb_groups(sa_class)

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

def run_IDEE(sa_class, path):
    """
    This function makes a serie of runs of the model IDEE.

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

def run_IDEE_multiproc(sa_class, path, nb=4):
    """
    Parallel version of the function run_IDEE.

    Input
        sa_class : SALib.util.problem.ProblemSpec
            the class of the library SAlib
        path : string
            the  directory where to save IDEE's outputs
        nb : integer
            number of CPU used
    """

    time_start = timer()
    nbmax = sa_class.samples.shape[0]
    name_dir = road.join(path, RAW_PATH)
    if not road.exists(name_dir):
        os.mkdir(name_dir)
                                                                         # run the model
    names = sa_class["names"]
                                                                         # parralel loop is here
    nb_samples = sa_class.samples.shape[0]
    args = [[dict(zip(names, sa_class.samples[n])), n, name_dir] for n in range(nb_samples)]

    with Pool(nb) as pool:
        for n in pool.imap(run_IDEE_multiproc_f, args):
            print("\n--- {:d} / {:d} ---".format(n+1, nbmax))

    print("  Total execution time = {:.1f} s".format(timer() - time_start))

def run_IDEE_multiproc_f(arg):
    """
    Function called into the parallel loop.

    Input
        arg : list (*)
            [params, number of sample, name of dir]
    """
    IDEE(arg[0], arg[1], arg[2])
    return arg[1]

def run_SA(sa_class, perc=[1, 99], rm_ex=False):
    """
    Run the sensitivity analysis of the sa_class.

    Input
        sa_class : SALib.util.problem.ProblemSpec
            the class of the library SAlib
        perc : list (float)
            percentiles of extremal values
        rm_ex : boolean
            if True, remove extremal values
    Output
        sa_class : SALib.util.problem.ProblemSpec
            the class of the library SAlib
    """
    if sa_class==None:
        print("There is no SA to do since there is no good points.")
        return None
                                                                         # copy class
    cc = ProblemSpec({
        "num_vars":sa_class["num_vars"],
        "names":sa_class["names"],
        "groups":sa_class["groups"],
        "bounds":sa_class["bounds"],
        "outputs":sa_class["outputs"]
    })
    cc = comp_nb_groups(cc)
                                                                         # get the parameters
    sorted_results = sa_class.results.copy()
    sorted_samples = sa_class.samples.copy()
    nbrows = sorted_results.shape[0]
                                                                         # remove extremal values
    if rm_ex:
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
    Nrem = nb_rows%(2*cc["nb_groups"]+2)
    print("  {:d} other samples to remove".format(Nrem))
    del_list = np.random.randint(low=0, high=nb_rows-1, size=Nrem)
    select_rows = [True]*nb_rows
    for i in del_list:
        select_rows[i] = False
    sorted_samples = sorted_samples[select_rows,:]
    sorted_results = sorted_results[select_rows,:]
                                                                         # update the class
    cc.set_samples(sorted_samples)
    cc.set_results(sorted_results)
    cc = comp_nb_samples(cc)
    cc["bad_array"] = np.zeros(cc["nb_samples"], dtype=np.bool8)
                                                                         # run the SA
    SA_time_start = timer()
    #Si = sobol.analyze(cc, cc.results, print_to_console=True)
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
    second_line = ""
    for var in sa_class["groups"]:
        second_line += var + ' '
                                                                         # the sample
    np.savetxt(
        fname=road.join(path, "sample.dat"),
        X=sa_class.samples,
        header=first_line + '\n' + second_line
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
        "groups":sa_class["groups"],
        "bounds":sa_class["bounds"],
        "outputs":sa_class["outputs"]
    })
    good_class = ProblemSpec({
        "num_vars":sa_class["num_vars"],
        "names":sa_class["names"],
        "groups":sa_class["groups"],
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
        bad_class = comp_nb_groups(bad_class)
        bad_class["bad_array"] = np.ones(bad_class["nb_samples"], dtype=np.bool8)
    else:
        print("  there is no bad samples")
        bad_class = None

    if good_s.shape[0] > 0:
        good_class.set_samples(good_s)
        good_class.set_results(good_r)
        good_class = comp_nb_samples(good_class)
        good_class = comp_nb_groups(good_class)
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
