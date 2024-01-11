"""
file name: _plot_func.py
language: Python 3
date: 12/12/2023
author: Hugo A. Martin
email: martin_hugo@ymail.com
description: plot funcitons
"""
                                                                         # --- imports ----------------------
from sa_idee._libs import *
from sa_idee._utils import *
                                                                         # --- macros -----------------------
T0, T1 = 50., 200
GAMMAW = 0.5
PHI0, PHI1 = -0.292, 0.469
DELTA, NU = 0.04, 3.
ALPHAC, KVC = 0.01, 0.5
DIV0, DIV1 = 0.0275, 0.4729
K_SCALE = 0.0045
MU, ETA = 1.7, 0.2
POW = 1./8
KAPPA0, KAPPA1 = 0.0397, 0.719
DELTANPOP, NPOPBAR = 0.05, 0.8*6.3
RB0 = 0.01

PHITAYLOR = 0.5
ETAR = 3.
ISTAR = 0.02
RSTAR = 0.02

Im, Ia = 2.5, 2.5
Gm, Ga = 2.5, 2.5
PIm, PIa = 19., 10.
LAMm, LAMa = 74., 6.
KAPPAm, KAPPAa = 19., 10.
OMEGAm, OMEGAa = 65., 7.
                                                                         # --- functions --------------------
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
    if c==None:
        print("nothing to plot")
        return 0

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
                                                                         # make the figure
    fig, axes = plt.subplots(nbrows, NUMCOLS, sharex=True, figsize=(A4W, 4.))
    is_bad = False
                                                                         # check whether it is a bad attractor
    is_bad = check_bad_attractor(data, lvars)
    if is_bad:
        ylim_year = -1
        #ylim_year = 100*data[:, lvars.index("inflation")]
        #ylim_year = np.arange(time.size)[ylim_year<-15]
        #ylim_year = ylim_year[0]
    else:
        ylim_year = -1

    k = 0
    while k<len(PLOT_LIST):
        try:
            var = plot_list_keys[k]
            ls = '-'
            if var=="capital" or var=="npop":
                if var=="capital":
                    ax = axes[-1,-1].twinx()
                    ls = '--'
                else:
                    ax = axes[-1,-1]
                raw = data[:, lvars.index(var)]
                mean = np.nan
                ax.set_ylabel(r"${}$".format(PLOT_LIST[var]))

            elif var=="debtratio":
                ax = axes[k//NUMCOLS, k%NUMCOLS]
                raw = 1./data[:, lvars.index(var)]
                mean, damp = comp_mean(raw, select)
                ax.set_ylabel(r"$1/d$")
                ax.plot(time, [1./NU]*time.size, color='C3', linestyle=':')

            else:
                ax = axes[k//NUMCOLS, k%NUMCOLS]
                raw = 100*data[:, lvars.index(var)]
                mean, damp = comp_mean(raw, select)
                ax.set_ylabel(r"${}$".format(PLOT_LIST[var])) # var.replace("_", "\n"))

            mean = mean if not is_bad else np.nan
            raw[ylim_year:] = raw[ylim_year]
            line1, line2, = ax.plot(time, raw, time, mean*np.ones(time.size), 
                linestyle=ls, label=r"${}$".format(PLOT_LIST[var]))

            if var=="npop":
                xlims, ylims = ax.get_xlim(), ax.get_ylim()
                line3, = ax.plot([0., 1.], [0., 1.], color="C0",
                    linestyle="--", label=r"$K$")
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)
                ax.legend(framealpha=1., loc=10, handles=[line1, line3])

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
    if badc==None or goodc==None:
        print("There is no map to plot since there is no bad or good point.")
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

def plot_param_aa(name, dval, start, stop):
    """
    Plot bounds

    Input
        name : string
            variable name
        dval : float
            default value
        start : float
            inf bound
        stop : float
            sup bound
    """
    print("{}: {:.6f} [{:.6f}, {:.6f}]".format(
        name,
        dval,
        min(start, stop),
        max(start, stop)
    ))

def plot_param_adot(radalpha=0.25, radkv=0.25, c=Gm/100, win=0.01, nb=5):
    """
    Plot the productivity growth.
    gdp growth rate
        https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG
    productivity growth rate
        https://www.researchgate.net/figure/Global-productivity-growth-since-1990-Annual-growth-rates_fig4_321381401

    Input
        radalpha : float
            the radius of alpha
        radkv : float
            the radius of the KV coefficient
        c : float
            central value where to mean
        win : float
            window width
        nb : integer
            number of lines
    """
    b_a = {"start":(1.-radalpha)*ALPHAC, "stop":(1.+radalpha)*ALPHAC}
    b_k = {"start":(1.-radkv)*KVC, "stop":(1.+radkv)*KVC}

    alphal = np.linspace(**b_a, num=nb)
    kvl = np.linspace(**b_k, num=nb)

    t = np.linspace(0., 1000, 1000)
    g = (Gm + Ga * np.exp(-t/T1) * np.sin(2.*np.pi*t/T0))/100
    select = (g>(c-win)) * (g<(c+win))

    fig, axes = plt.subplots(nrows=3)
    axes[2].plot(t, g)

    means = []
    for alpha in alphal:
        for kv in kvl:
            dat = alpha + kv*g
            axes[0].plot(
                g,
                dat,
                color="k",
                linestyle="-",
            )
            axes[1].plot(
                t,
                dat,
                color="k",
                linestyle="-",
            )
            means.append(np.mean(dat[select]))

    dat = ALPHAC + KVC*g
    axes[0].plot(
        g,
        ALPHAC + KVC*g,
        color="r",
        linestyle="-"
    )
    axes[1].plot(
        t,
        ALPHAC + KVC*g,
        color="r",
        linestyle="-"
    )
    tip_val = np.mean(dat[select])

    ylims = axes[0].get_ylim()
    axes[0].fill_between([c-win, c+win], ylims[0], ylims[1], color="k", alpha=0.25, zorder=100)
    axes[0].set_xlabel(r"growth rate ($g$)")
    axes[0].set_ylabel(r"productivity growth ($\dot{a}/a$)")
    axes[1].set_xlabel(r"time ($t$)")
    axes[1].set_ylabel(r"productivity growth ($\dot{a}/a$)")
    axes[2].set_ylabel(r"growth rate ($g$)")
    axes[2].set_xlabel(r"time ($t$)")

    for ax in axes[:-1]:
        ax.plot(
            list(ax.get_xlim()),
            [0., 0.],
            color="gray",
            linestyle="--"
        )

    print("\nwindow size = {:.1f} %".format((np.max(means)-np.min(means))/tip_val*100))
    plot_param_aa("alpha", ALPHAC, **b_a)
    plot_param_aa("kaldoor-verdon", KVC, **b_k)

    plt.show()
    plt.close(fig)

def plot_param_all():
    """
    Plot all param functions.
    """
    print("\n  Productivity growth")
    plot_param_adot()
    print("\n  Central Bank interest rate")
    plot_param_CB_interest_rate()
    print("\n  Delta nu")
    plot_param_deltanu()
    print("\n  Dividends")
    plot_param_dividends()
    print("\n  Gamma")
    plot_param_Gamma()
    print("\n  Gammaw")
    plot_param_gammaw()
    print("\n  Inflation")
    plot_param_inflation()
    print("\n  Investment")
    plot_param_investment()
    print("\n  Phillips")
    plot_param_phillips()
    print("\n  Population")
    plot_param_population()

def plot_param_CB_interest_rate(rad0=0.4, rad1=0.4, rad2=0.4, rad3=0.3, c0=Im/100, win0=0.01, c1=62.25, win1=5, nb0=5, nb1=5):
    """
    plot central bank interest rate
    inflation
        https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=1W
    interest rate
        https://data.worldbank.org/indicator/FR.INR.LEND?locations=US-CN-IN-GB&view=chart 

    Input
    """
    a = {"start":(1.-rad0)*PHITAYLOR, "stop":(1.+rad0)*PHITAYLOR}
    b = {"start":(1.-rad1)*ISTAR, "stop":(1.+rad1)*ISTAR}
    cc = {"start":(1.-rad2)*RSTAR, "stop":(1.+rad2)*RSTAR}
    d = {"start":(1.-rad3)*ETAR, "stop":(1.+rad3)*ETAR}
    dt = 1.

    phil = np.linspace(**a, num=nb0)
    istarl = np.linspace(**b, num=nb0)
    rstarl = np.linspace(**cc, num=nb0)
    etarl = np.linspace(**d, num=nb1)

    t = np.linspace(0., 1000, 4000)
    select1 = (t>(c1-win1)) * (t<(c1+win1))
    i = (Im + Ia * np.exp(-t/T1) * np.sin(2.*np.pi*t/T0))/100
    select0 = (i>(c0-win0)) * (i<(c0+win0))

    fig, axes = plt.subplots(nrows=3)
    axes[2].plot(t, i)

    means0 = []
    means1 = []
    for phi in phil:
        for istar in istarl:
            for rstar in rstarl:
                rcb = np.maximum(0., rstar + i + phi*(i - istar))
                axes[0].plot(i, rcb, color="k", linestyle="-")
                axes[1].plot(t, rcb, color="k", linestyle="-")
                means0.append(np.mean(rcb[select0]))
    
    rcb = np.maximum(0., RSTAR + i + PHITAYLOR*(i - ISTAR))
    tmaxrcb = t[np.argmax(rcb[select1])]
    axes[0].plot(i, rcb, color="r", linestyle="-")
    axes[1].plot(t, rcb, color="r", linestyle="-")
    for etar in etarl:
        rb = np.zeros(t.size)
        rb[0] = RB0
        for n, tt in enumerate(t[:-1]):
                                                                         # time step is one year here
            rb[n+1] = rb[n] + dt/etar * (rcb[n] - rb[n])
        axes[1].plot(t, rb, color="gray", linestyle="-", alpha=0.5)
        means1.append(
            abs(tmaxrcb - t[np.argmax(rb[select1])])
        )

    for n, tt in enumerate(t[:-1]):
        rb[n+1] = rb[n] + dt/ETAR * (rcb[n] - rb[n])
    axes[1].plot(t, rb, color="r", linestyle="--")
    tip_val0 = np.mean(rcb[select0])
    tip_val1 = abs(tmaxrcb - t[np.argmax(rb[select1])])

    ylims = axes[0].get_ylim()
    axes[0].fill_between([(c0-win0), (c0+win0)], ylims[0], ylims[1], color="k", alpha=0.25, zorder=100)
    ylims = axes[1].get_ylim()
    axes[1].fill_between([(c1-win1), (c1+win1)], ylims[0], ylims[1], color="k", alpha=0.25, zorder=100)

    axes[0].set_xlabel(r"inflation ($i$)")
    axes[0].set_ylabel(r"central bank interest rate ($r_{cb}$)")
    axes[1].set_xlabel(r"time ($t$)")
    axes[1].set_ylabel(r"central bank interest rate ($r_{cb}$)")
    axes[2].set_ylabel(r"inflation ($i$)")
    axes[2].set_xlabel(r"time ($t$)")

    print("\nwindow size = {:.1f} %".format((np.max(means0)-np.min(means0))/tip_val0*100))
    plot_param_aa("phistar", PHITAYLOR, **a)
    plot_param_aa("istar", ISTAR, **b)
    plot_param_aa("rstar", RSTAR, **cc)

    print("\nwindow size = {:.1f} %".format((np.max(means1)-np.min(means1))/tip_val1*100))
    plot_param_aa("etar", ETAR, **d)

    plt.show()
    plt.close(fig)

def plot_param_deltanu(rad0=0.125, rad1=0.13, c=KAPPAm/100, win=0.02, nb=5):
    """
    Plot delta and nu parameters.

    gdp growth rate
        https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG

    Some data

    investment constant 2015 USD
        https://data.worldbank.org/indicator/NE.GDI.TOTL.KD?contextual=default&end=2022&locations=CN-JP-US-IN-EU&start=2000&view=chart

        c = [1., 3.36, 5.82, 6.8]
        us = [3., 3., 3.86, 4.44]
        eu = [2.57, 2.68, 2.81, 3.49]
        j = [1.2, 1.2, 1.2, 1.2]
        i = [0.1, 0.6, 0.7, 1.12]

    gdp constant 2015 USD
        https://data.worldbank.org/indicator/NY.GDP.MKTP.KD?e

        gdpc = [3.6, 7.55, 11.06, 16.33]
        gdpus = [14.5, 16.38, 18.21, 20.93]
        gdpeu = [11.74, 12.9, 13.55, 15.28]
        gdpj = [4, 4, 4.44, 4.5]
        gdpi = [0.8, 1.54, 2.1, 2.96]

    debt ratio
        https://wolfstreet.com/2017/11/22/private-sector-debt-implode-next-us-eurozone-japan-china-or-canada/

        dc = [1.1, 1.5, 2.1, 2.9]
        dus = [1.7, 1.5, 1.5, 1.6]
        deu = [1.5, 1.65, 1.62, 1.3]
        dj = [1.6, 1.68, 1.6, 1.5]

    sumI, sumGDP, prctI, sumd, prctd = [], [], [], [], []
    for n in range(4):
        sumI.append(c[n] + us[n] + eu[n] + j[n] + i[n])
        sumGDP.append(gdpc[n] + gdpus[n] + gdpeu[n] + gdpj[n] + gdpi[n])
        sumd.append(dc[n] + dus[n] + deu[n] + dj[n])
        prctI.append(sumI[n] / sumGDP[n])
        prctd.append(sumd[n] / (sumGDP[n] - gdpi[n]))

    results
        sumI = [7.87, 10.839999999999998, 14.389999999999999, 17.05]
        sumGDP = [34.64, 42.37, 49.36000000000001, 60.0]
        prctI = [0.22719399538106236, 0.2558413972150106, 0.29153160453808746, 0.2841666666666667]

    Input
        rad0: float
        rad1 : float
    """

    a = {"start":(1.-rad0)*DELTA, "stop":(1.+rad0)*DELTA}
    b = {"start":(1.-rad1)*NU, "stop":(1.+rad1)*NU}

    deltal = np.linspace(**a, num=nb)
    nul = np.linspace(**b, num=nb)

    t = np.linspace(0., 1000, 1000)
    kappa = (KAPPAm + KAPPAa * np.exp(-t/T1) * np.sin(2.*np.pi*t/T0))/100
    select = (kappa>(c-win)) * (kappa<(c+win))

    fig, axes = plt.subplots(nrows=3)
    axes[2].plot(t, kappa)

    means = []
    for delta in deltal:
        for nu in nul:
            g = kappa / nu - delta
            axes[0].plot(
                kappa,
                g,
                linestyle="-",
                color="k"
            )
            axes[1].plot(
                t,
                g,
                linestyle="-",
                color="k"
            )
            means.append(np.mean(g[select]))
    g = kappa / NU - DELTA
    axes[0].plot(
        kappa,
        g,
        linestyle="-",
        color="r"
    )
    axes[1].plot(
        t,
        g,
        linestyle="-",
        color="r"
    )
    tip_val = np.mean(g[select])

    ylims = axes[0].get_ylim()
    axes[0].fill_between([(c-win), (c+win)], ylims[0], ylims[1], color="k", alpha=0.25, zorder=100)

    axes[0].set_xlabel(r"investment ratio (\kappa$)")
    axes[0].set_ylabel(r"growth rate ($g$)")
    axes[1].set_xlabel(r"time ($t$)")
    axes[1].set_ylabel(r"growth rate ($g$)")
    axes[2].set_ylabel(r"investment ratio ($\kappa$)")
    axes[2].set_xlabel(r"time ($t$)")

    print("\nwindow size = {:.1f} %".format((np.max(means)-np.min(means))/tip_val*100))
    plot_param_aa("delta", DELTA, **a)
    plot_param_aa("nu", NU, **b)

    plt.show()
    plt.close(fig)

def plot_param_dividends(rad0=0.25, rad1=0.25, c=PIm/NU/100, win=0.005, nb=5):
    """
    Plot dividend functions
    dividends
        https://revenusetdividendes.com/les-entreprises-francaises-sont-elles-vraiment-plus-genereuses-en-dividendes-en-europe/
    profit
        http://dln.jaipuria.ac.in:8080/jspui/bitstream/123456789/2865/1/MGI%20Global%20Competition_Full%20Report_Sep%202015.pdf

    Input
        rad0 : float
            radius of const param
        rad1 : float
            radius of slope param
        nb : integer
            number of points
    """
    MAXB, MAXT = 0.1, 0.3
    PROFITTODIV = 0.45
    print("dividends are set to approx 45% of profits.")

    a = {"start":(1.-rad0)*DIV0, "stop":(1.+rad0)*DIV0}
    b = {"start":(1.-rad1)*DIV1, "stop":(1.+rad1)*DIV1}

    div0l = np.linspace(**a, num=nb)
    div1l = np.linspace(**b, num=nb)

    t = np.linspace(0., 1000, 1000)
    smallpik = (PIm + PIa * np.exp(-t/T1) * np.sin(2.*np.pi*t/T0))/100/NU
    div_that_should_be = np.clip(PROFITTODIV * smallpik, 0., 0.3)*NU

    select = (smallpik>(c-win)) * (smallpik<(c+win))

    fig, axes = plt.subplots(nrows=3)
    axes[2].plot(t, NU*smallpik)

    means = []
    for div0 in div0l:
        for div1 in div1l:
            div = np.clip(
                div0 + div1*smallpik,
                0.,
                0.3
            )*NU
            axes[0].plot(
                smallpik*NU,
                div,
                linestyle="-",
                color="k"
            )
            axes[1].plot(
                t,
                div,
                linestyle="-",
                color="k"
            )
            means.append(np.mean(div[select]))
    div = np.clip(
        DIV0 + DIV1*smallpik,
        0.,
        0.3
    )*NU
    axes[0].plot(
        smallpik*NU,
        div,
        linestyle="-",
        color="r"
    )
    axes[1].plot(
        t,
        div,
        linestyle="-",
        color="r"
    )
    axes[1].plot(
        t,
        div_that_should_be,
        linestyle="--",
        color="r"
    )
    tip_val = np.mean(div[select])

    ylims = axes[0].get_ylim()
    axes[0].fill_between([(c-win)*NU, (c+win)*NU], ylims[0], ylims[1], color="k", alpha=0.25, zorder=100)

    axes[0].set_xlabel(r"profit to GDP ($\Pi/pY$)")
    axes[0].set_ylabel(r"dividends rate (to GDP) ($\Delta$)")
    axes[1].set_xlabel(r"time ($t$)")
    axes[1].set_ylabel(r"dividends rate (to GDP) ($\Delta$)")
    axes[2].set_ylabel(r"profit to GDP ($\Pi/pY$)")
    axes[2].set_xlabel(r"time ($t$)")

    print("\nwindow size = {:.1f} %".format((np.max(means)-np.min(means))/tip_val*100))
    plot_param_aa("div0", DIV0, **a)
    plot_param_aa("div1", DIV1, **b)

    plt.show()
    plt.close(fig)

def plot_param_Gamma(rad=0.255, c=0.96, win=0.03, nb=5):
    """
    Plot te Gamma function.

    Input
        rad : float
            radius of k_scale \in [(1+-rad)*K_SCALE]
    """
    b = {"start":max(K_SCALE/4, (1.-rad)*K_SCALE), "stop":(1.+rad)*K_SCALE}

    ks_list = np.linspace(**b, num=nb)
    tcdebtratio = np.linspace(0., 0.99999*NU, 500)
    select = (tcdebtratio>(c-win)) * (tcdebtratio<(c+win))

    fig, ax = plt.subplots()
    means = []
    for k_scale in ks_list:
        gammad = 1. - np.exp(
            -k_scale*(tcdebtratio**2)/(NU**2 - tcdebtratio**2)
        )
        ax.plot(
            tcdebtratio/NU,
            gammad,
            linestyle="-",
            color="k"
        )
        means.append(np.mean(gammad[select]))
    gammad = 1. - np.exp(
        -K_SCALE*(tcdebtratio**2)/(NU**2 - tcdebtratio**2)
    )
    ax.plot(
        tcdebtratio/NU,
        gammad,
        linestyle="-",
        color="r"
    )
    tip_val = np.mean(gammad[select])

    ylims = ax.get_ylim()
    ax.fill_between([(c-win), (c+win)], ylims[0], ylims[1], color="k", alpha=0.25, zorder=100)
    ax.set_xlabel(r"debt to capital ratio ($D/pK/\nu$)")
    ax.set_ylabel(r"Gamma function ($\Gamma$)")

    print("\nwindow size = {:.1f} %".format((np.max(means)-np.min(means))/tip_val*100))
    plot_param_aa("k_scale", K_SCALE, **b)

    plt.show()
    plt.close(fig)

def plot_param_gammaw(rad=0.1, c=LAMm/100, win=0.02, nb=5):
    """
    Plot the influence of gammaw on Phillips curve.

    wage growth
        https://www.ilo.org/asia/media-centre/news/WCMS_651039/lang--ja/index.htm
    employment 15 - 64
        https://en.wikipedia.org/wiki/List_of_sovereign_states_by_employment_rate

    Input
        rad : float
            radius of gammaw param \in [(1+-rad)*GAMMAW]
        nb : integer
            number of lines
    """
    lams = np.linspace(0.5, 0.7, 100)

    t = np.linspace(0., 1000, 1000)
    lams = (LAMm + LAMa * np.exp(-t/T1) * np.sin(2.*np.pi*t/T0))/100
    select = (lams>(c-win)) * (lams<(c+win))

    I = np.linspace(-0.01, 0.04, 7)
    b_a = {"start":(1.-rad)*GAMMAW, "stop":(1.+rad)*GAMMAW}
    gammaw = np.linspace(**b_a, num=nb)

    fig, axes = plt.subplots(nrows=3)
    axes[2].plot(t, lams)

    means = []
    for n, i in enumerate(I):
        for gamma in gammaw:
            dat = PHI0 + PHI1*lams + gamma*i
            axes[0].plot(
                lams,
                dat,
                linestyle="-",
                color="k"
            )
            axes[1].plot(
                t,
                dat,
                linestyle="-",
                color="k"
            )
            if n==I.size-1:
                means.append(np.mean(dat[select]))
        axes[0].plot(
            lams,
            PHI0 + PHI1*lams + GAMMAW*i,
            linestyle="-",
            color="r"
        )
        axes[1].plot(
            t,
            PHI0 + PHI1*lams + GAMMAW*i,
            linestyle="-",
            color="r"
        )
        if n==I.size-1:
            tip_val = np.mean(dat[select])
           
    ylims = axes[0].get_ylim()
    axes[0].fill_between([(c-win), (c+win)], ylims[0], ylims[1], color="k", alpha=0.25, zorder=100)

    axes[0].set_xlabel(r"employment ($\lambda$)")
    axes[0].set_ylabel(r"Phillips curve ($\varphi$)")
    axes[1].set_xlabel(r"time ($t$)")
    axes[1].set_ylabel(r"Phillips curve ($\varphi$)")
    axes[2].set_ylabel(r"employment ($\lambda$)")
    axes[2].set_xlabel(r"time ($t$)")

    print("\nwindow size = {:.1f} %".format((np.max(means)-np.min(means))/tip_val*100))
    plot_param_aa("gammaw", GAMMAW, **b_a)

    plt.show()
    plt.close(fig)

def plot_param_inflation(rad0=0.02, rad1=0.15, c=OMEGAm/100, win=0.015, nb=5):
    """
    plot inflation depending on eta and mu

    inflation
        https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=1W
    wage
        https://ourworldindata.org/grapher/labor-share-of-gdp?tab=chart&time=2004..2020
    profit ratio
        http://dln.jaipuria.ac.in:8080/jspui/bitstream/123456789/2865/1/MGI%20Global%20Competition_Full%20Report_Sep%202015.pdf

    """
    t = np.linspace(0., 1000, 1000)
    omega = (OMEGAm + OMEGAa * np.exp(-t/T1) * np.sin(2.*np.pi*t/T0))/100
    smallpik = (PIm + PIa * np.exp(-t/T1) * np.sin(2.*np.pi*t/T0))/100/NU
    select = (omega>(c-win)) * (omega<(c+win))

    a = {"start":(1.-rad0)*MU, "stop":(1.+rad0)*MU}
    b = {"start":(1.-rad1)*ETA, "stop":(1.+rad1)*ETA}

    mul = np.linspace(**a, num=nb)
    etal = np.linspace(**b, num=nb)

    fig, axes = plt.subplots(nrows=4)
    axes[2].plot(
        t,
        omega,
        linestyle="-",
        color="k"
    )
    axes[3].plot(
        t,
        smallpik,
        linestyle="-",
        color="k"
    )
    means = []
    for mu in mul:
        for eta in etal:
            i = eta * ( (mu + smallpik) * omega - 1.)
            axes[0].plot(
                omega,
                i,
                linestyle="-",
                color="k"
            )
            axes[1].plot(
                t, #omega,
                i,
                linestyle="-",
                color="k"
            )
            means.append(np.mean(i[select]))

    i = ETA * ( (MU + smallpik) * omega - 1.)
    axes[0].plot(
        omega,
        i,
        linestyle="-",
        color="r"
    )
    axes[1].plot(
        t, #omega,
        i,
        linestyle="-",
        color="r"
    )
    tip_val = np.mean(i[select])
 
    ylims = axes[0].get_ylim()
    axes[0].fill_between([(c-win), (c+win)], ylims[0], ylims[1], color="k", alpha=0.25, zorder=100)

    axes[0].set_xlabel(r"wage share ($\omega$)")
    axes[0].set_ylabel(r"inflation ($i$)")
    axes[1].set_ylabel(r"inflation ($i$)")
    axes[1].set_xlabel(r"time ($t$)")
    axes[2].set_ylabel(r"wage share ($\omega$)")
    axes[2].set_xlabel(r"time ($t$)")
    axes[3].set_ylabel(r"profit to capital ratio ($\Pi / p K$)")
    axes[3].set_xlabel(r"time ($t$)")
    for ax in axes[:-2]:
        ax.plot(
            list(ax.get_xlim()),
            [0., 0.],
            color="gray",
            linestyle="--"
        )

    print("\nwindow size = {:.1f} %".format((np.max(means)-np.min(means))/tip_val*100))
    plot_param_aa("mu", MU, **a)
    plot_param_aa("eta", ETA, **b)

    plt.show()
    plt.close(fig)

def plot_param_investment(rad0=0.25, rad1=0.25, c=KAPPAm/100, win=0.02, nb=5):
    """
    Plot investment functions

    investment constant 2015 USD
        https://data.worldbank.org/indicator/NE.GDI.TOTL.KD?contextual=default&end=2022&locations=CN-JP-US-IN-EU&start=2000&view=chart

        prctI = [0.22719399538106236, 0.2558413972150106, 0.29153160453808746, 0

    gdp growth rate
        https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG
    profit
        http://dln.jaipuria.ac.in:8080/jspui/bitstream/123456789/2865/1/MGI%20Global%20Competition_Full%20Report_Sep%202015.pdf

    Input
        rad0 : float
            radius of const param
        rad1 : float
            radius of slope param
        nb : integer
            number of points
    """
    KAPPAMIN, KAPPAMAX = 0., 0.5
    MAXB, MAXT = 0.1, 0.3

    a = {"start":(1.-rad0)*KAPPA0, "stop":(1.+rad0)*KAPPA0}
    b = {"start":(1.-rad1)*KAPPA1, "stop":(1.+rad1)*KAPPA1}

    kappa0l = np.linspace(**a, num=nb)
    kappa1l = np.linspace(**b, num=nb)

    tcdebtratio = np.linspace(0., 1.05*NU, 1000)
    t = np.linspace(0., 1000, 1000)
    smallpi = (PIm + PIa * np.exp(-t/T1) * np.sin(2.*np.pi*t/T0))/100
    select = (smallpi>(c-win)) * (smallpi<(c+win))

    fig, axes = plt.subplots(nrows=5)
    axes[2].plot(t, [0.]*t.size, color="gray", linestyle="--")
    axes[3].plot(t, smallpi)
    axes[4].plot(t, tcdebtratio/NU*100)
    axes[4].plot([t[0], t[-1]], [100., 100.])

    means = []
    for kappa0 in kappa0l:
        for kappa1 in kappa1l:

            kappapi = kappa0 + kappa1*smallpi
            tmp = np.maximum(0., 1. - tcdebtratio/NU)
            kappapi = kappapi*(tmp)**POW
            kappapi = np.clip(kappapi, KAPPAMIN, KAPPAMAX)

            axes[0].plot(
                smallpi,
                kappapi,
                color="k",
                linestyle="-",
            )
            axes[1].plot(
                t,
                kappapi,
                color="k",
                linestyle="-",
            )
            axes[2].plot(
                t,
                kappapi/NU - DELTA,
                color="k",
                linestyle="-",
            )
            means.append(np.mean(kappapi[select]))

    kappapi = KAPPA0 + KAPPA1*smallpi
    tmp = np.maximum(0., 1. - tcdebtratio/NU)
    kappapi = kappapi*(tmp)**POW
    kappapi = np.clip(kappapi, KAPPAMIN, KAPPAMAX)
    axes[0].plot(
        smallpi,
        kappapi,
        color="r",
        linestyle="-",
    )    
    axes[1].plot(
        t,
        kappapi,
        color="r",
        linestyle="-",
    )    
    axes[2].plot(
        t,
        kappapi/NU - DELTA,
        color="r",
        linestyle="-",
    )    
    tip_val = np.mean(kappapi[select])

    ylims = axes[0].get_ylim()
    axes[0].fill_between([(c-win), (c+win)], ylims[0], ylims[1], color="k", alpha=0.25, zorder=100)

    axes[0].set_xlabel(r"profit to GDP ($\pi$)")
    axes[0].set_ylabel(r"investment rate (to GDP) ($\kappa$)")
    axes[1].set_xlabel(r"time $t$)")
    axes[1].set_ylabel(r"investment rate (to GDP) ($\kappa$)")
    axes[2].set_xlabel(r"time $t$)")
    axes[2].set_ylabel(r"growth rate ($g$)")
    axes[3].set_ylabel(r"profit to GDP ($\pi$)")
    axes[3].set_xlabel(r"time ($t$)")
    axes[4].set_ylabel(r"debt to capital ratio ($d/\nu$)")
    axes[4].set_xlabel(r"time ($t$)")

    print("\nwindow size = {:.1f} %".format((np.max(means)-np.min(means))/tip_val*100))
    plot_param_aa("kappa0", KAPPA0, **a)
    plot_param_aa("kappa1", KAPPA1, **b)

    plt.show()
    plt.close(fig)

def plot_param_phillips(rad0=0.006, rad1=0.035, c=LAMm/100, win=0.005, nb=5):
    """
    Plot the phillips curve.

    employment 15 - 64
        https://en.wikipedia.org/wiki/List_of_sovereign_states_by_employment_rate
    wage growth
        https://www.ilo.org/asia/media-centre/news/WCMS_651039/lang--ja/index.htm

    Input
        rad0 : float
            radius of the const param \in [(1+-rad0)*PHI0]
        rad1 : float
            radius of the slope param \in [(1+-rad1)*PHI1]
        nb : integer
            number of lines
    """
    t = np.linspace(0., 1000, 1000)
    lams = (LAMm + LAMa * np.exp(-t/T1) * np.sin(2.*np.pi*t/T0))/100
    select = (lams>(c-win)) * (lams<(c+win))

    a = {"start":(1.-rad0)*PHI0, "stop":(1.+rad0)*PHI0}
    b = {"start":max(0., (1.-rad1)*PHI1), "stop":(1.+rad1)*PHI1}

    fig, axes = plt.subplots(nrows=3)
    axes[2].plot(
        t,
        lams
    )

    phi0 = np.linspace(**a, num=nb)
    phi1 = np.linspace(**b, num=nb)
    means = []
    for n, p0 in enumerate(phi0):
        for m, p1 in enumerate(phi1):
            phi = phi0[n] + phi1[m]*lams
            axes[0].plot(
                lams,
                phi,
                color="k",
                linestyle="-",
            )
            axes[1].plot(
                t,
                phi,
                color="k",
                linestyle="-",
            )
            means.append(np.mean(phi[select]))
    axes[0].plot(
        lams,
        PHI0 + PHI1*lams,
        color="r"
    )
    axes[1].plot(
        t,
        PHI0 + PHI1*lams,
        color="r",
        linestyle="-",
    )
    tip_val = np.mean(phi[select])

    axes[0].set_ylabel(r"Phillips curve ($\varphi$)")
    axes[0].set_xlabel(r"employment ($\lambda$)")
    axes[1].set_ylabel(r"Phillips curve ($\varphi$)")
    axes[1].set_xlabel(r"time ($t$)")
    axes[2].set_xlabel(r" (time $t$)")
    axes[2].set_ylabel(r"employment ($\lambda$)")

    for ax in axes[:-1]:
        ax.plot(
            list(ax.get_xlim()),
            [0., 0.],
            color="gray",
            linestyle="--"
        )

    ylims = axes[0].get_ylim()
    axes[0].fill_between([(c-win), (c+win)], ylims[0], ylims[1], color="k", alpha=0.25, zorder=100)

    print("\nwindow size = {:.1f} %".format((np.max(means)-np.min(means))/tip_val*100))
    plot_param_aa("phi0", PHI0, **a)
    plot_param_aa("phi1", PHI1, **b)

    plt.show()
    plt.close(fig)

def plot_param_population(rad0=0.6, rad1=0.075, c=2100, win=10, nb=5):
    """
    Plot the population trajectories.
    15-56 ans
        https://population.un.org/wpp/Graphs/DemographicProfiles/Line/900
    public sector size
        https://en.wikipedia.org/wiki/List_of_countries_by_public_sector_size

    Input
        rad0 : float
            radius of the deltanpop \in [(1+-rad0)*PHI0]
        rad1 : float
            radius of the npopmax \in [(1+-rad1)*PHI1]
        nb : integer
            number of lines
    """
    a = {"start":(1.-rad0)*DELTANPOP, "stop":(1.+rad0)*DELTANPOP}
    b = {"start":(1.-rad1)*NPOPBAR, "stop":(1.+rad1)*NPOPBAR}
    deltanpop = np.linspace(**a, num=nb)
    npopbar = np.linspace(**b, num=nb)

    fig, ax = plt.subplots()
    k = 0
    means = []
    for delta in deltanpop:
        for npop in npopbar:
            pop = integrate_pop(delta, npop)
            ax.plot(pop[:,0], pop[:,1], color="k", linestyle="-")
            select = (pop[:,0]>(c-win)) * (pop[:,0]<(c+win))
            means.append(np.mean(pop[select, 1]))
            k+=1

    pop = integrate_pop(DELTANPOP, NPOPBAR)
    ax.plot(pop[:,0], pop[:,1], color="r", linestyle="-")
    select = (pop[:,0]>(c-win)) * (pop[:,0]<(c+win))
    tip_val = np.mean(pop[select, 1])

    ax.set_xlabel(r"time")
    ax.set_ylabel(r"population ($N$)")
    ylims = ax.get_ylim()
    ax.fill_between([(c-win), (c+win)], ylims[0], ylims[1], color="k", alpha=0.25, zorder=100)

    print("\nwindow size = {:.1f} %".format((np.max(means)-np.min(means))/tip_val*100))
    plot_param_aa("delta n pop", DELTANPOP, **a)
    plot_param_aa("n pop ax", NPOPBAR, **b)

    plt.show()
    plt.close(fig)

def plot_sa_class(sa_class, path, ylims=[-0.1, 1.1], figure_name="sa.pdf", show=False):
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
    if sa_class==None:
        print("nothing to plot")
        return 0

    axes = sa_class.plot()
    axes[0,0].set_ylim(ylims[0], ylims[1])
    for row in axes:
        for col in row:
            xlims = col.get_xlim()
            col.plot([xlims[0], xlims[1]], [0., 0.])
            col.plot([xlims[0], xlims[1]], [1., 1.])

    plt.tight_layout(**PADS)

    fig = plt.gcf()
    perso_savefig(fig, path, figure_name, show)

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
