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

def plot_phillips(rad0=0.5, rad1=0.9, nb=10):
    """
    Plot the phillips curve.

    Input
        rad0 : float
            radius of the const param \in [(1+-rad0)*PHI0]
        rad1 : float
            radius of the slope param \in [(1+-rad1)*PHI1]
        nb : integer
            number of lines
    """
    OMEGA = np.linspace(0., 1., 1000)
    PHI0, PHI1 = -0.292, 0.469

    fig, ax = plt.subplots()
    ax.plot(OMEGA, [0.]*OMEGA.size, "--", color="gray")
    phi0 = np.linspace((1.-rad0)*PHI0, (1.+rad0)*PHI0, nb)
    phi1 = np.linspace((1.-rad1)*PHI1, (1.+rad1)*PHI1, nb)
    for n, p0 in enumerate(phi0):
        for m, p1 in enumerate(phi1):
            phi = phi0[n] + phi1[m]*OMEGA
            ax.plot(
                OMEGA,
                phi,
                color="k",
                linestyle="-",
            )
    ax.plot(OMEGA, PHI0 + PHI1*OMEGA, color="r")
    ax.fill_between(OMEGA, -0.2, 0.2, color="k", alpha=0.25, zorder=100)
    ax.fill_between([0.5, 0.8], -0.3, 0.3, color="k", alpha=0.25, zorder=100)
    ax.set_ylabel(r"$\varphi$")
    ax.set_xlabel(r"$\omega$")
    plt.show()
    plt.close(fig)

def plot_gammaw(rad=0.25, nb=4):
    """
    Plot the influence of gammaw on inflation.

    Input
        rad : float
            radius of gammaw param \in [(1+-rad)*GAMMAW]
        nb : integer
            number of lines
    """
    GAMMAW = 0.5
    PHI0, PHI1 = -0.292, 0.469
    OMEGA = np.linspace(0., 1., 1000)
    I = np.linspace(-0.5, 0.5, 1000)
    gammaw = np.linspace((1.-rad)*GAMMAW, (1.+rad)*GAMMAW, nb)
    OMEGA, I = np.meshgrid(OMEGA, I)

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    for gamma in gammaw:
        ax.plot_surface(
            X=OMEGA,
            Y=I,
            Z=PHI0 + PHI1*OMEGA + gamma*I,
            cmap=cm.coolwarm
        )
    ax.plot_surface(
        X=OMEGA,
        Y=I,
        Z=PHI0 + PHI1*OMEGA + GAMMAW*I,
        color="grey"
    )
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$i$")
    ax.set_zlabel(r"$\phi$")
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
