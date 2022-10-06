import os
import numpy as np
import configparser
import sys
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import SymLogNorm
import matplotlib.mlab
import scipy.io as sio
import pandas as pd
import scipy.optimize as optim
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy import fft, arange, signal
import matplotlib as mpl


import argparse


mpl.rcParams["lines.linewidth"] = 2.5
mpl.rcParams["savefig.bbox"] = "tight"
# mpl.rcParams["figure.figsize"] = (16, 5)
mpl.rcParams["font.size"] = 18
mpl.rcParams["legend.frameon"] = False


parser = argparse.ArgumentParser(description="graphparameters")
parser.add_argument("--pattern", type=int)
parser.add_argument("--cearth", type=float)
parser.add_argument("--impulse", type=int)
parser.add_argument("--baseline", type=str)
parser.add_argument("--year", type=int)

args = parser.parse_args()


# os.chdir('/Users/erikchavez/Documents/Papers/Economic_Policy/C-T-dynamic-ODEs/')

# INPUT PARAMETERS


def model(pulse, year, cearth=0.3916, baseline="rcp60co2eqv3.csv"):
    # heat capacity, incoming radiation
    # Earth heat capacity
    # cearth = 0.3916
    # Incoming radiation
    Q0 = 342.5

    # land fraction and albedo
    # Fraction of land on the planet
    p = 0.3
    # land albedo
    alphaland = 0.28

    # outgoing radiation linearized
    kappa = 1.74
    Tkappa = 154

    # Ocean albedo parameters
    Talphaocean_low = 219
    Talphaocean_high = 299
    alphaocean_max = 0.843
    alphaocean_min = 0.254

    # CO2 radiative forcing
    # Greenhouse effect parameter
    B = 5.35
    # CO2 params. C0 is the reference C02 level
    C0 = 280

    # ocean carbon pumps
    # Solubility dependence on temperature (value from Fowler et al)
    bP = 0.029
    # Biopump dependence on temperature (Value from Fowler)
    bB = 0.069
    # Ocean carbon pump modulation parameter
    cod = 2.2

    # timescale and reference temperature (from Fowler)
    # timescale
    tauc = 30
    # Temperature reference
    T0 = 288

    # Coc0 ocean carbon depending on depth
    coc0 = 280

    # CO2 uptake by vegetation

    # lower and upper G thresholds
    Cbio_low = 150
    Cbio_high = 750

    # vegetation carbon uptake temperatures
    Thigh = 307.15
    Tlow = 286.15
    Topt1 = 290.15
    Topt2 = 302.15
    acc = 8

    # lower and upper bounds of opt and viable temp
    Tbiopt1_low = Topt1
    Tbiopt1_high = Topt1 + 5
    Tbiolow_low = Tlow
    Tbiolow_high = Tlow + 5

    # vegetation growth parameters
    wa = 0.015
    vegcover = 0.4

    # Volcanism and atmospheric expansion (Hogg 2008 and LeQuere 2015)
    V = 2.028

    # Anthropogenic carbon
    # Switch to take anthropogenic emissions
    sa = 1
    # Anthropogenic emissions (zero or one)
    # csvname = baseline+'.csv'
    Can = pd.read_csv(baseline)
    #Can = pd.read_csv("Et-sim2.csv")
    # times2co2eq
    # rcp85co2eq.csv
    #Ca = Can[(Can["YEARS"] > 1899) & (Can["YEARS"] < 2201)]
    #Ca = Can[(Can["YEARS"] > 1799) & (Can["YEARS"] < 2501)]
    Ca = Can[(Can["YEARS"] > 1799) & (Can["YEARS"] < 2801)]
    # Ca1 = Can[(Can["YEARS"] > 1799) & (Can["YEARS"] < 2801)]
    #Ca["YEARS"] = np.arange(start=0,stop=401,step=1)
    #Ca = Ca.pd.DataFrame()
    Ca = Ca["CO2EQ"]
    #Ca = Ca - 286.76808
    Ca = Ca - 281.69873
    Ca = Ca.to_numpy()

    Ca[year-1800] += pulse

    tspan = len(Ca)

    #Ce = np.arange(401)
    #Ce = np.arange(601)
    Ce = np.arange(tspan) * 1.0
    # np.min(Ca)
    for i in range(len(Ce)):
        if i == 0:
            Ce[i] = 0
        else:
            Ce[i] = Ca[i] - Ca[i-1]

    Cebis = np.arange(tspan) * 1.0
    # np.min(Ca)
    for i in range(len(Cebis)):
        if i == 0:
            Cebis[i] = 0
        else:
            Cebis[i] = max(Ca[i] - Ca[i-1], 0)

    Cc = np.arange(tspan) * 1.0
    # np.min(Ca)
    for i in range(len(Cc)):
        if i == 0:
            Cc[i] = 0
        else:
            Cc[i] = sum(Cebis[0:i])

    # FUNCTIONS

    # Anthropogenic carbon fitting with cubic spline
    t_val = np.linspace(0, tspan-1, tspan)

    def Yem(t):
        t_points = t_val
        em_points = Ca

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    def Yam(t):
        t_points = t_val
        em_points = Cebis

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    def Ycm(t):
        t_points = t_val
        em_points = Cc

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    # Ocean albedo

    def alphaocean(T):
        if T < Talphaocean_low:
            return alphaocean_max
        elif T < Talphaocean_high:
            return alphaocean_max + (alphaocean_min - alphaocean_max) / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
        else:  # so T is higher
            return alphaocean_min

    # Fraction of ocean covered by ice

    def fracseaice(T):
        if T < Talphaocean_low:
            return 1
        elif T < Talphaocean_high:
            return 1 - 1 / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
        else:  # so T is higher
            return 0

    # Vegetation growth function

    def veggrowth(T):
        if T < Tlow:
            return 0
        if (T >= Tlow) and (T < Topt1):
            return acc / (Topt1 - Tlow) * (T - Tlow)
        if (T >= Topt1) and (T <= Topt2):
            return acc
        if (T > Topt2) and (T < Thigh):
            # return acc
            return acc / (Topt2 - Thigh) * (T - Thigh)
        if T >= Thigh:
            # return acc
            return 0

    # T_values = np.linspace(280, 315, 201)
    # plt.plot(T_values, [veggrowth(val) for val in T_values])
    # plt.tick_params(axis='both', which='major', labelsize=13)
    # plt.xlabel('Temperature (K)',fontsize = 14);
    # plt.ylabel('Vegetation growth',fontsize = 14);
    # plt.grid(linestyle=':')
    # veggrowth(286.6181299517094)

    # ramp function of lower optimum temperature
    def Tbioptlow(Cc):
        if Cc < Cbio_low:
            return Tbiopt1_low
        elif Cc < Cbio_high:
            return Tbiopt1_low + (Tbiopt1_high - Tbiopt1_low) / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
            # return 1 - 2 / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
        else:  # so Cc is higher
            return Tbiopt1_high
            # return -1

    Tbioptlow = np.vectorize(Tbioptlow)

    # ramp function of percentage of vegetation carbon lost
    Vecar_min = 0
    Vecar_max = 5/12

    def Bioloss(Cc):
        if Cc < Cbio_low:
            return Vecar_min
        elif Cc < Cbio_high:
            return Vecar_min + (Vecar_max - Vecar_min) / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
        else:  # so Cc is higher
            return Vecar_max
            # return -1

    Bioloss = np.vectorize(Bioloss)

    # ramp function of vegetation carbon lost (ppm)
    C0v = 1000
    VC_min = 0
    VC_max = 5/12 * C0v

    def BioCloss(Cc):
        if Cc < Cbio_low:
            return VC_min
        elif Cc < Cbio_high:
            return VC_min + (VC_max - VC_min) / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
        else:  # so Cc is higher
            return VC_max
            # return -1

    BioCloss = np.vectorize(BioCloss)

    # evolution of the lower optimum temperature with cumulative emission scenario
    Toptmodulation = [Tbioptlow(val) for val in Cc]
    Toptmod = np.float_(Toptmodulation)

    def Tvegoptlow(t):
        t_points = t_val
        em_points = Toptmod

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    # evolution of the percentage of the stock of vegetation carbon lost
    Coptmodulation = [Bioloss(val) for val in Cc]
    Coptmod = np.float_(Coptmodulation)

    def Cvegoptlow(t):
        t_points = t_val
        em_points = Coptmod

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    # evolution of the vegetation carbon lost
    VCoptmodulation = [BioCloss(val) for val in Cc]
    VCoptmod = np.float_(VCoptmodulation)

    def VCvegoptlow(t):
        t_points = t_val
        em_points = VCoptmod

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    # Tbiolow
    Tbiolow_low = Tlow
    Tbiolow_high = Tlow + 5

    def Tbiolow(Cc):
        if Cc < Cbio_low:
            return Tbiolow_low
        elif Cc < Cbio_high:
            return Tbiolow_low + (Tbiolow_high - Tbiolow_low) / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
            # return 1 - 2 / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
        else:  # so Cc is higher
            return Tbiolow_high
            # return -1

    Tbiolow = np.vectorize(Tbiolow)

    # evolution of the lower viable temperature with cumulative emission scenario
    Tlowmodulation = [Tbiolow(val) for val in Cc]
    Tlowmod = np.float_(Tlowmodulation)

    def Tveglow(t):
        t_points = t_val
        em_points = Tlowmod

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    # Vegetation growth function
    def veggrowthdyn(T, t):
        if T < Tveglow(t):
            return 0
        if (T >= Tveglow(t)) and (T < Tvegoptlow(t)):
            return acc / (Tvegoptlow(t) - Tveglow(t)) * (T - Tveglow(t))
        if (T >= Tvegoptlow(t)) and (T <= Topt2):
            return acc
        if (T > Topt2) and (T < Thigh):
            # return acc
            return acc / (Topt2 - Thigh) * (T - Thigh)
        if T > Thigh:
            # return acc
            return 0

    # Incoming radiation modified by albedo

    def Ri(T):
        return 1/cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean(T)))

    # Outgoing radiation modified by greenhouse effect

    def Ro(T, C):
        return 1/cearth * (kappa * (T - Tkappa) - B * np.log(C / C0))

    # vegetation carbon flux

    def vegfluxdyn(T, C, t):
        return wa * C * vegcover * veggrowthdyn(T, t)

    # ocean carbon fluxes

    def oceanatmphysflux(T):
        return 1 / tauc * (coc0 * (np.exp(-bP * (T - T0))))

    def oceanbioflux(T):
        return 1/tauc * (coc0 * (np.exp(bB * (T - T0))))

    def oceanatmcorrflux(C):
        return 1 / tauc * (- cod * C)

    # MODEL EQUATIONS

    def dydt(t, y):
        T = y[0]
        C = y[1]
        #Cveg = y[3]

        dT = Ri(T)
        dT -= Ro(T, C)

        dC = V
        # anthropogenic emissions from Ca spline                                                # volcanism
        dC += Yam(t) * sa
        # dC += Ca * sa                                       # added for bif diagrams
        # dC -= wa * C * vegcover * veggrowth(T)             # carbon uptake by vegetation
        #dC -= vegflux(T, C, t)
        dC -= vegfluxdyn(T, C, t)
        # physical solubility into ocean * fraction of ice-free ocean
        dC += oceanatmphysflux(T) * (1 - fracseaice(T))
        # dC += oceanbioflux(T,t) * (1 - fracseaice(T))      # biological pump flux * fraction sea ice
        # biological pump flux * fraction sea ice
        dC += oceanbioflux(T) * (1 - fracseaice(T))
        dC += oceanatmcorrflux(C) * (1 - fracseaice(T)
                                     )    # correction parameter

        return dT, dC

    # Integrate the ODE

    sa = 1
    Ts = 286.45
    Cs = 269

    length = 1000
    init = [Ts, Cs]
    t_eval = np.linspace(0, tspan, length)
    sol = solve_ivp(dydt, t_eval[[0, -1]], init,
                    t_eval=t_eval, method='RK45', max_step=1)
    #sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='BDF')

    # Extract values of temperature and C02
    Tv = sol.y[0, :]
    Cv = sol.y[1, :]
    tv = sol.t

    # Fixed points
    # print('Tp = {:.1f}'.format(Tv[-1]))
    # print('Cp = {:.1f}'.format(Cv[-1]))

    Tvmid = Tv - Ts
    # Cvmid = Cv - Cs

    # Tvmean = np.mean(Tv)
    # Tvmin = np.min(Tv)
    # Tvmax = np.max(Tv)

    # Total atmospheric carbon
    Ct = Cv + VCvegoptlow(t_eval)
    G = Ycm(t_eval)
    modelsol = [tv+1800, Tvmid, Ct, G, Tvmid]
    return modelsol


# plot from 1800 to 2400
colors = ['blue', 'green', 'red', 'gold', 'cyan',
          'magenta', 'yellow', 'salmon', 'grey', 'black']
# titles = ['Impulse Response of Temperature Anomaly T',
#           'Impulse Response of Carbon Concentration Dynamics C']
ylabels = ['Temperature (K)', 'Carbon (ppm)',
           'Emission (Gtc)', 'Degree (Celsius)']
# fig, axs = plt.subplots(len(selected_index),1, figsize = (3*(len(selected_index)),20), dpi = 200)

# ceartharray = np.array((0.3725, 0.3916, 15))
# pulsearray = np.arange(0, 100, 1)
baselinearray = ["carbonvoid.csv",  "rcp85co2eqv3.csv", "rcp60co2eqv3.csv",
                 "rcp45co2eqv3.csv", "rcp30co2eqv3.csv", "rcp00co2eqv3.csv"]
# yeararray = np.array((1801, 2010))
# Figure_Dir = "./nonlinearCarbon/figure/pulse_average/"
Figure_Dir = "./figure/NC_PulseExp/"

pattern = args.pattern
cearth = args.cearth
impulse = args.impulse
baseline = args.baseline
year = args.year

modelsolBase = model(pulse=0, year=year,
                     cearth=cearth, baseline=baseline)

fig, axs = plt.subplots(2, 1, figsize=(9, 9))

modelsol = model(impulse, year, cearth, baseline)

print(baseline, cearth, impulse)

if pattern == 0:
    titles = ["Impulse Response of Temperature Anomaly",
              "Impulse Response of Carbon Concentration Dynamics"]
if pattern == 1:
    titles = ["Impulse Response of Temperature Anomaly per Gtc",
              "Impulse Response of Carbon Concentration Dynamics per Gtc"]
if pattern == 2:
    titles = ["Impulse Response of Temperature Anomaly per Ttc",
              "Impulse Response of Carbon Concentration Dynamics per Ttc"]

for j in range(2):
    if pattern == 0:
        axs[j].plot(modelsol[0], modelsol[j+1]-modelsolBase[j+1], color=colors[j %
                    len(colors)], label=f"cearth={cearth},pulse={impulse}")
    if pattern == 1:
        axs[j].plot(modelsol[0], (modelsol[j+1]-modelsolBase[j+1])/impulse, color=colors[j %
                    len(colors)], label=f"cearth={cearth},pulse={impulse}")
    if pattern == 2:
        axs[j].plot(modelsol[0], (modelsol[j+1]-modelsolBase[j+1])/1000, color=colors[j %
                    len(colors)], label=f"cearth={cearth},pulse={impulse}")
    axs[j].set_ylabel(ylabels[j])
    axs[j].set_title(titles[j])
    if j == 1:
        axs[j].set_xlabel('Year')
    axs[j].legend(loc='lower right')

plt.tight_layout()
# plt.savefig(Figure_Dir+"Baseline="+baseline+",cearth" +
#             str(cearth)+",year"+str(year)+",pulse="+str(pulse)+".pdf")
if pattern == 0:
    plt.savefig(Figure_Dir+"Baseline="+baseline+",cearth=" +
                str(cearth)+",year="+str(year)+",pulse="+str(impulse)+",2IRF.png")
if pattern == 1:
    plt.savefig(Figure_Dir+"Baseline="+baseline+",cearth=" +
                str(cearth)+",year="+str(year)+",pulse="+str(impulse)+",2IRF,per.png")
if pattern == 2:
    plt.savefig(Figure_Dir+"Baseline="+baseline+",cearth=" +
                str(cearth)+",year="+str(year)+",pulse="+str(impulse)+",2IRF,Tera.png")
# plt.show()
plt.close()


# # plot from 1800 to 2400
# colors = ['blue', 'green', 'red', 'gold', 'cyan',
#           'magenta', 'yellow', 'salmon', 'grey', 'black']
# titles = ['Impulse Response of Temperature Anomaly per Gtc',
#           'Impulse Response of Carbon Concentration Dynamics per Gtc']
# ylabels = ['Temperature (K)', 'Carbon (ppm)',
#            'Emission (Gtc)', 'Degree (Celsius)']
# # fig, axs = plt.subplots(len(selected_index),1, figsize = (3*(len(selected_index)),20), dpi = 200)

# ceartharray = np.array((0.3725, 0.3916, 15))
# pulsearray = np.arange(0, 100, 1)
# baselinearray = ["carbonvoid.csv",  "rcp85co2eqv3.csv", "rcp60co2eqv3.csv",
#                  "rcp45co2eqv3.csv", "rcp30co2eqv3.csv", "rcp00co2eqv3.csv"]
# yeararray = np.array((1801, 2010))
# # Figure_Dir = "./nonlinearCarbon/figure/pulse_average/"
# Figure_Dir = "./figure/NC_PulseExp/"
# # year = 1801

# # cearth = 0.3725
# # pulse = 1
# # baseline = "rcp60co2eqv3.csv"

# # model(pulse, year, cearth, baseline)

# for year in yeararray:

#     for baseline in baselinearray:

#         for cearth in ceartharray:

#             modelsolBase = model(pulse=0, year=year,
#                                  cearth=cearth, baseline=baseline)

#             for pulse in pulsearray:

#                 fig, axs = plt.subplots(2, 1, figsize=(3*4, 20))

#                 modelsol = model(pulse, year, cearth, baseline)

#                 print(baseline, cearth, pulse)

#                 for j in range(2):

#                     axs[j].plot(modelsol[0], (modelsol[j+1]-modelsolBase[j+1])/pulse, color=colors[j %
#                                 len(colors)], label=f"cearth={cearth},pulse={pulse}")
#                     axs[j].set_ylabel(ylabels[j])
#                     axs[j].set_title(titles[j])
#                     axs[j].set_xlabel('Year')
#                     axs[j].legend(loc='lower right')

#                 plt.tight_layout()
#                 # plt.savefig(Figure_Dir+"Baseline="+baseline+",cearth" +
#                 #             str(cearth)+",year"+str(year)+",pulse="+str(pulse)+".pdf")
#                 plt.savefig(Figure_Dir+"Baseline="+baseline+",cearth=" +
#                             str(cearth)+",year="+str(year)+",pulse="+str(pulse)+",2IRF,per.png")
#                 # plt.show()
#                 plt.close()


# # plot from 1800 to 2400
# colors = ['blue', 'green', 'red', 'gold', 'cyan',
#           'magenta', 'yellow', 'salmon', 'grey', 'black']
# titles = ['Impulse Response of Temperature Anomaly per Ttc',
#           'Impulse Response of Carbon Concentration Dynamics per Ttc']
# ylabels = ['Temperature (K)', 'Carbon (ppm)',
#            'Emission (Gtc)', 'Degree (Celsius)']
# # fig, axs = plt.subplots(len(selected_index),1, figsize = (3*(len(selected_index)),20), dpi = 200)

# ceartharray = np.array((0.3725, 0.3916, 15))
# pulsearray = np.arange(0, 100, 1)
# baselinearray = ["carbonvoid.csv",  "rcp85co2eqv3.csv", "rcp60co2eqv3.csv",
#                  "rcp45co2eqv3.csv", "rcp30co2eqv3.csv", "rcp00co2eqv3.csv"]
# yeararray = np.array((1801, 2010))
# # Figure_Dir = "./nonlinearCarbon/figure/pulse_average/"
# Figure_Dir = "./figure/NC_PulseExp/"

# for year in yeararray:

#     for baseline in baselinearray:

#         for cearth in ceartharray:

#             modelsolBase = model(pulse=0, year=year,
#                                  cearth=cearth, baseline=baseline)

#             for pulse in pulsearray:

#                 fig, axs = plt.subplots(2, 1, figsize=(3*4, 20))

#                 modelsol = model(pulse, year, cearth, baseline)

#                 print(baseline, cearth, pulse)

#                 for j in range(2):

#                     axs[j].plot(modelsol[0], (modelsol[j+1]-modelsolBase[j+1])/1000, color=colors[j %
#                                 len(colors)], label=f"cearth={cearth},pulse={pulse}")
#                     axs[j].set_ylabel(ylabels[j])
#                     axs[j].set_title(titles[j])
#                     axs[j].set_xlabel('Year')
#                     axs[j].legend(loc='lower right')

#                 plt.tight_layout()
#                 # plt.savefig(Figure_Dir+"Baseline="+baseline+",cearth" +
#                 #             str(cearth)+",year"+str(year)+",pulse="+str(pulse)+".pdf")
#                 plt.savefig(Figure_Dir+"Baseline="+baseline+",cearth=" +
#                             str(cearth)+",year="+str(year)+",pulse="+str(pulse)+",2IRF,Tera.png")
#                 # plt.show()
#                 plt.close()
