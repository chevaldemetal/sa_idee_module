"""
file name: _utils.py
language: Python 3
date: 12/12/2023
author: Hugo A. Martin
email: martin_hugo@ymail.com
description: Some utilities functions.
"""
                                                                         # --- imports ----------------------
from sa_idee._libs import *
                                                                         # --- functions --------------------
def check_bad_attractor(data, lvars):
    """
    This function checks whether the simulation converges toward the bad attractor.

    Input
        data : numpy.ndarray (float)
            the array containing the simulation's data
        lvars : list (string)
            the array of variables' name
    Output
        is_bad : boolean
            True wheter it converges toard the bad attractor
    """
    is_bad = False
    indices = [lvars.index("capital"), lvars.index("omega"), lvars.index("lambda")]
    for ind in indices:
        raw = data[:, ind]
        is_bad = is_bad or (raw<POURCENT).any() or np.isnan(raw).any()

    return is_bad

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
    new[:,0] = data[:,lvars.index("investment")] / data[:,lvars.index("gdp")]
    data = np.hstack((data, new))
                                                                         # append dividends ratio
    lvars.append("dividends_ratio")
    new = np.zeros((nbrows,1))
    new[:,0] = np.clip(data[:,lvars.index("smallpi")] - data[:,lvars.index("pir")] / data[:,lvars.index("gdp")] / data[:,lvars.index("price")], 0., 0.3)
    data = np.hstack((data, new))
                                                                         # append wage growth rate
    lvars.append("wage_growth")
    new = np.zeros((nbrows,1))
    new[:,0] = comp_growth(data[:,lvars.index("wage")], nb)
    data = np.hstack((data, new))
                                                                         # append productivity growth
    lvars.append("productivity_growth")
    new = np.zeros((nbrows,1))
    new[:,0] = comp_growth(data[:,lvars.index("productivity")], nb)
    data = np.hstack((data, new))
                                                                         # append population growth
    lvars.append("population_growth")
    new = np.zeros((nbrows,1))
    new[:,0] = comp_growth(data[:,lvars.index("npop")], nb)
    data = np.hstack((data, new))
    return lvars, data

def integrate_pop(deltanpop, npopbar):
    """
    Integrate population trajectories.

    Input
        deltanpop : float
            growth rate
        npopbar : float
            upper bound of population
    Ouput
        sol : numpy.ndarray (float)
            solution array
    """
    t0 = 2015
    y0 = 0.8*5.5
    dt = 1./12
    t_bound = 3000

    fun = lambda t, y: y*deltanpop*(1. - y/npopbar)

    P = RK45(fun, t0, [y0], t_bound, first_step=dt)
    P.t_old = P.t

    N = [y0]
    time = [t0]
    while P.status=="running":
        P.step()
        interp = P.dense_output()

        for t in np.arange(P.t_old, P.t, dt):
            y = interp(t)

            N.append(y[0])
            time.append(t)

    sol = np.column_stack((time, N))

    return sol
