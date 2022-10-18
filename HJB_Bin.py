from supportfunctions import finiteDiff
import SolveLinSys
from scipy.interpolate import RegularGridInterpolator
from scipy import fft, arange, signal
from scipy import interpolate
from scipy.optimize import curve_fit
import scipy.optimize as optim
import scipy.io as sio
import matplotlib.mlab
from matplotlib.colors import SymLogNorm
from matplotlib import rcParams
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import solve_ivp
import pickle
import pandas as pd
import numpy as np
import sys
import os
import argparse


parser = argparse.ArgumentParser(description="values")

parser.add_argument("--maxiter",type=int,default=5000)
parser.add_argument("--fraction",type=float,default=0.1)
parser.add_argument("--epsilon",type=float)

parser.add_argument("--simutime",type=int,default=5000)


parser.add_argument("--hXarr",nargs='+',type=float, default=(0.4, 4.0, 40.0))
parser.add_argument("--Xminarr",nargs='+',type=float, default=(1e-8, 0.0, -5.5))
parser.add_argument("--Xmaxarr",nargs='+',type=float, default=(9.0, 4.0, 0.0))

parser.add_argument("--cearth",type=float,default=0.3916)
parser.add_argument("--tauc",type=float,default=30)

parser.add_argument("--filename",type=str)

args = parser.parse_args()

print(args.Xmaxarr)
print(args.Xminarr)
print(args.hXarr)
sys.path.append("../src/")
rcParams["figure.figsize"] = (8, 5)
rcParams["savefig.bbox"] = 'tight'

# print("load complete")
def PDESolver(stateSpace, A, B1, B2, B3, C1, C2, C3, D, v0, 
              ε = 1, tol = -10):                                              
                                                                                 

    A = A.reshape(-1,1,order = 'F')                                         
    B = np.hstack([B1.reshape(-1,1,order = 'F'),B2.reshape(-1,1,order = 'F'),B3.reshape(-1,1,order='F')])
    C = np.hstack([C1.reshape(-1,1,order = 'F'),C2.reshape(-1,1,order = 'F'),C3.reshape(-1,1,order='F')])
    D = D.reshape(-1,1,order = 'F')                                         
    v0 = v0.reshape(-1,1,order = 'F')                                       
    out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0, ε, tol)           

    return out  


def model(cearth=0.3916,tauc = 30):

    #############################################
    ##########Climate Change Part################
    #############################################
    #############################################
    

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

    # Ocean albedo

    # def alphaocean(T):
    #     if T < Talphaocean_low:
    #         return alphaocean_max
    #     elif T < Talphaocean_high:
    #         return alphaocean_max + (alphaocean_min - alphaocean_max) / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    #     else:  # so T is higher
    #         return alphaocean_min

    def alphaocean(T):
        """T, matrix, (nT, nC, nF)"""
        temp = np.zeros(T.shape)
        temp[T < Talphaocean_low] = alphaocean_max
        temp[(T >= Talphaocean_low) & (T < Talphaocean_high)] = alphaocean_max + (alphaocean_min - alphaocean_max) / \
            (Talphaocean_high - Talphaocean_low) * \
            (T[(T >= Talphaocean_low) & (T < Talphaocean_high)] - Talphaocean_low)
        temp[T >= Talphaocean_high] = alphaocean_min

        return temp

    # Fraction of ocean covered by ice

    # def fracseaice(T):
    #     if T < Talphaocean_low:
    #         return 1
    #     elif T < Talphaocean_high:
    #         return 1 - 1 / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    #     else:  # so T is higher
    #         return 0

    def fracseaice(T):

        temp = np.zeros(T.shape)
        temp[T < Talphaocean_low] = 1
        temp[(T >= Talphaocean_low) & (T < Talphaocean_high)] = 1 - 1 / (Talphaocean_high -
                                                                         Talphaocean_low) * (T[(T >= Talphaocean_low) & (T < Talphaocean_high)] - Talphaocean_low)
        temp[T >= Talphaocean_high] = 0

        return temp

    # Vegetation growth function

    # def veggrowth(T):
    #     if T < Tlow:
    #         return 0
    #     if (T >= Tlow) and (T < Topt1):
    #         return acc / (Topt1 - Tlow) * (T - Tlow)
    #     if (T >= Topt1) and (T <= Topt2):
    #         return acc
    #     if (T > Topt2) and (T < Thigh):
    #         # return acc
    #         return acc / (Topt2 - Thigh) * (T - Thigh)
    #     if T >= Thigh:
    #         # return acc
    #         return 0

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




    # Tbiolow
    Tbiolow_low = Tlow
    Tbiolow_high = Tlow + 5



    # def Tveglow(t):
    #     t_points = t_val
    #     em_points = Tlowmod

    #     tck = interpolate.splrep(t_points, em_points)
    #     return interpolate.splev(t, tck)

    def Tveglow(G):

        temp = np.zeros(G.shape)

        temp[G <= Cbio_low] = Tbiolow_low
        temp[(G > Cbio_low) & (G < Cbio_high)] = Tbiolow_low + \
            (Tbiolow_high - Tbiolow_low) / \
            (Cbio_high - Cbio_low) * \
            (G[(G > Cbio_low) & (G < Cbio_high)] - Cbio_low)
        temp[G >= Cbio_high] = Tbiolow_high

        return temp

    # ramp function of vegetation carbon lost (ppm)
    C0v = 1000
    VC_min = 0
    VC_max = 5/12 * C0v


    # def Tvegoptlow(t):
    #     t_points = t_val
    #     em_points = Toptmod

    #     tck = interpolate.splrep(t_points, em_points)
    #     return interpolate.splev(t, tck)

    def Tvegoptlow(G):

        temp = np.zeros(G.shape)

        temp[G <= Cbio_low] = VC_min
        temp[(G > Cbio_low) & (G < Cbio_high)] = VC_min + \
            (VC_max - VC_min) / \
            (Cbio_high - Cbio_low) * \
            (G[(G > Cbio_low) & (G < Cbio_high)] - Cbio_low)
        temp[G >= Cbio_high] = VC_max

        return temp

    # # Vegetation growth function
    # def veggrowthdyn(T, t):
    #     if T < Tveglow(t):
    #         return 0
    #     if (T >= Tveglow(t)) and (T < Tvegoptlow(t)):
    #         return acc / (Tvegoptlow(t) - Tveglow(t)) * (T - Tveglow(t))
    #     if (T >= Tvegoptlow(t)) and (T <= Topt2):
    #         return acc
    #     if (T > Topt2) and (T < Thigh):
    #         # return acc
    #         return acc / (Topt2 - Thigh) * (T - Thigh)
    #     if T > Thigh:
    #         # return acc
    #         return 0

    # Vegetation growth function: Vectorized

    def veggrowthdyn(T, G):

        temp = np.zeros(T.shape)

        temp[T < Tveglow(G)] = 0

        temp[(T >= Tveglow(G)) & (T < Tvegoptlow(G))] = acc / (Tvegoptlow(G)[(T >= Tveglow(G)) & (T < Tvegoptlow(G))] - Tveglow(G)[(T >= Tveglow(G)) & (T < Tvegoptlow(G))]) * \
            (T[(T >= Tveglow(G)) & (T < Tvegoptlow(G))] - Tveglow(G)[(T >= Tveglow(G)) & (T < Tvegoptlow(G))])

        temp[(T >= Tvegoptlow(G)) & (T <= Topt2)] = acc

        temp[(T >= Topt2) & (T < Thigh)] = acc / (Topt2 - Thigh) * \
            (T[(T >= Topt2) & (T < Thigh)] - Thigh)

        temp[T > Thigh] = 0

        return temp

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



    # Integrate the ODE

    sa = 1

    #############################################
    ########Economic Model Part###################
    #############################################
    #############################################

    # Economic paramaters
    gamma_1 = 1.7675 / 10000.
    gamma_2 = 2 * 0.0022
    delta = 0.01
    eta = 0.032

    # State variable
    # Temperature anomaly, in celsius
    T_min  = args.Xminarr[0]
    T_max  = args.Xmaxarr[0] # 
    hT     = args.hXarr[0]
    T_grid = np.arange(T_min, T_max + hT, hT)

    # atmospheric carbon concentration, in gigaton
    C_min  = args.Xminarr[1]
    C_max  = args.Xmaxarr[1]
    hC     = args.hXarr[1]
    C_grid = np.arange(C_min, C_max + hC, hC)

    # F, Sa in the notes, accumulative anthropogenic carbon, in gigaton, since 1800
    F_min = args.Xminarr[2] # 10. avaoid 
    F_max = args.Xmaxarr[2] # 2500 x2.13 gm # # on hold -> 4000 / 2.13 ppm
    hF = args.hXarr[2]
    F_grid = np.arange(F_min, F_max + hF, hF)

    # meshgrid
    (T_mat, C_mat, F_mat) = np.meshgrid(T_grid, C_grid, F_grid, indexing="ij")
    stateSpace = np.hstack([
        T_mat.reshape(-1, 1, order="F"),
        C_mat.reshape(-1, 1, order="F"),
        F_mat.reshape(-1, 1, order="F")
    ])

    print(T_mat.shape)
    To = 282.87  # Mean with no anthropogenic carbon emissions, in Fᵒ



    # v0 = pickle.load(open("data_35.0_6603", "rb"))["v0"]
    v0 = - eta * T_mat - eta * F_mat
    # v0 =  delta * eta * np.log(delta /4 * (9000/2.13 - F_mat)) + (eta - 1) * gamma_2 * T_mat / cearth * (B * np.log(C_mat/ C0) + kappa * (T_mat + To - Tkappa))

    dG = gamma_1 + gamma_2 * T_mat
    epsilon = args.epsilon
    count = 0
    error = 1.
    tol = 1e-8
    max_iter = args.maxiter
    fraction = args.fraction

    while error > tol and count < max_iter:

        dvdT = finiteDiff(v0, 0, 1, hT)
        dvdTT = finiteDiff(v0, 0, 2, hT)
        dvdC = finiteDiff(v0, 1, 1, hC)
    #     dvdC[dvdC >= - 1e-16] = - 1e-16
        dvdCC = finiteDiff(v0, 1, 2, hC)
        dvdF = finiteDiff(v0, 2, 1, hF)
        dvdFF = finiteDiff(v0, 2, 2, hF)

        Ca = - eta * delta / (dvdC + dvdF)

        Ca[Ca <= 1e-32] = 1e-32

        if count >= 1:
            Ca = Ca * fraction + Ca_star * (1 - fraction)

    #     Ca = np.ones(T_mat.shape)
        A = - delta * np.ones(T_mat.shape)
        B1 = Ri(T_mat + To) - Ro(T_mat + To, C_mat)
        B2 = V
        B2 += Ca * sa
        # B2 -= wa * C_mat * veggrowth(T_mat + To)
        B2 -= wa * C_mat * veggrowthdyn(T_mat + To, F_mat)
        B2 += oceanatmphysflux(T_mat + To) * (1 - fracseaice(T_mat + To))
        B2 += oceanbioflux(T_mat + To) * \
            (1 - fracseaice(T_mat + To))
        B2 += oceanatmcorrflux(C_mat) * (1 - fracseaice(T_mat + To))
        B3 = Ca
        C1 = np.zeros(T_mat.shape)
        C2 = np.zeros(T_mat.shape)
        C3 = np.zeros(T_mat.shape)
        D = eta * delta * np.log(Ca) + (eta - 1) * \
            dG * B1  # relation between versions?

        out = PDESolver(stateSpace, A, B1, B2, B3, C1, C2, C3, D, v0, epsilon)
        v = out[2].reshape(v0.shape, order="F")

        rhs_error = A * v0 + B1 * dvdT + B2 * dvdC + B3 * \
            dvdF + C1 * dvdTT + C2 * dvdCC + C3 * dvdFF + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0)/epsilon))

        error = lhs_error
        v0 = v
        Ca_star = Ca
        count += 1
        if count%1000==0:
            print("Iteration: %s;\t False Transient Error: %s;\t PDE Error: %s\t" %
                (count, lhs_error, rhs_error),flush=True)

    print("Total iteration: %s;\t LHS Error: %s;\t RHS Error %s\t" %
          (count, lhs_error, rhs_error),flush=True)

    res = {
    "v0": v0,
    "T": T_grid,
    "C": C_grid,
    "F": F_grid,
    "Ca": Ca,
    }
    Data_Dir = "./data/"+args.filename+"/"
    
    os.makedirs(Data_Dir, exist_ok=True)

    pickle.dump(res, open(Data_Dir+f"data_{cearth}_{tauc}_{args.maxiter}_{args.fraction}_{args.epsilon}_{args.Xminarr}_{args.Xmaxarr}_{args.hXarr}", "wb"))


    return T_grid,C_grid,F_grid,Ca,cearth,tauc

def simulation(T_grid,C_grid,F_grid,Ca,cearth=0.3916,tauc = 30):
    To = 282.87  # Mean with no anthropogenic carbon emissions, in Fᵒ
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


    t_max = args.simutime
    dt = 1/12
    sa = 1  # , Gigaton per year
    gridpoints = (T_grid, C_grid, F_grid)   
    Ca_func = RegularGridInterpolator(gridpoints, Ca)

    # T_0 = To + min(T_grid)
    # C_0 = 275.5
    # F_0 = min(F_grid) #(870 - 580) / 2.13 # total cumulated, as of now, preindustrial with Fo

    T_0 = To + 1.1
    C_0 = 275.5
    F_0 = (870 - 580) / 2.13

    def get_e(x):
        return Ca_func([x[0] - To, x[1], x[2]])

    # Ocean albedo
    def alphaocean(T):
        if T < Talphaocean_low:
            return alphaocean_max
        elif T < Talphaocean_high:
            return alphaocean_max + (alphaocean_min - alphaocean_max) / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
        else:  # so T is higher
            return alphaocean_min

    C0v = 1000
    VC_min = 0
    VC_max = 5/12 * C0v
    
    def Tvegoptlow(Cc):
        if Cc < Cbio_low:
            return VC_min
        elif Cc < Cbio_high:
            return VC_min + (VC_max - VC_min) / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
        else:  # so Cc is higher
            return VC_max
            # return -1

    def Tveglow(Cc):

        if Cc < Cbio_low:
            return Tbiolow_low
        elif Cc < Cbio_high:
            return Tbiolow_low + (Tbiolow_high - Tbiolow_low) / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
            # return 1 - 2 / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
        else:  # so Cc is higher
            return Tbiolow_high
            # return -1
        
    # Vegetation growth function
    def veggrowthdyn(T, G):
        if T < Tveglow(G):
            return 0
        if (T >= Tveglow(G)) and (T < Tvegoptlow(G)):
            return acc / (Tvegoptlow(G) - Tveglow(G)) * (T - Tveglow(G))
        if (T >= Tvegoptlow(G)) and (T <= Topt2):
            return acc
        if (T > Topt2) and (T < Thigh):
            # return acc
            return acc / (Topt2 - Thigh) * (T - Thigh)
        if T > Thigh:
            # return acc
            return 0


    def oceanatmphysflux(T):
        return 1 / tauc * (coc0 * (np.exp(-bP * (T - T0))))

    def fracseaice(T):
        if T < Talphaocean_low:
            return 1
        elif T < Talphaocean_high:
            return 1 - 1 / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
        else:  # so T is higher
            return 0

    # def biopump(Cc):
    #     if Cc < Cbio_low:
    #         return 1
    #     elif Cc < Cbio_high:
    #         return 1 - 1 / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
    #     else: 
    #         return 0


    def oceanbioflux(T):
        return 1/tauc * (coc0 * (np.exp(bB * (T - T0))))

    def oceanatmcorrflux(C):
        return 1 / tauc * (- cod * C)


    def mu_T(x):
        Ri_t = 1 / cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean(x[0])))
        Ro_t = 1 / cearth * (kappa * (x[0] - Tkappa) -  B * np.log(x[1] / C0))
        return Ri_t - Ro_t

    def mu_C(x):
        Ca_t = Ca_func([x[0] - To, x[1], x[2]])
        dC = V
        dC += Ca_t * sa
        dC -= wa * x[1] * vegcover * veggrowthdyn(x[0],x[2])
        dC += oceanatmphysflux(x[0]) * (1 - fracseaice(x[0]))
        dC += oceanbioflux(x[0]) * (1 - fracseaice(x[0]))
        dC += oceanatmcorrflux(x[1]) * (1 - fracseaice(x[0]))
        return dC

    def mu_Sa(x):
        return Ca_func([x[0] - To, x[1], x[2]])

    years  = np.arange(0, t_max + dt, dt)
    pers   = len(years)

    hist      = np.zeros([pers, 3])
    e_hist    = np.zeros([pers])


    for tm in range(pers):
        if tm == 0:
            # initial points
            hist[0,:] = [T_0, C_0, F_0] # logL
            e_hist[0] = get_e(hist[0, :])

        else:
            # other periods
            e_hist[tm] = get_e(hist[tm-1,:])

            hist[tm,0] = max(hist[tm-1,0] + mu_T(hist[tm-1,:]) * dt, To + min(T_grid))
            hist[tm,1] = hist[tm-1,1] + mu_C(hist[tm-1,:]) * dt
            hist[tm,2] = hist[tm-1,2] + mu_Sa(hist[tm-1,:]) * dt

    plt.subplots(1,3, figsize=(24,5))
    plt.subplot(131)
    plt.plot(years, hist[:, 0] - To)
    plt.xlabel("Years")
    plt.title("Temperature anomaly")
    plt.subplot(132)
    plt.plot(years, hist[:, 1])
    plt.xlabel("Years")
    plt.title("Atmospheric carbon concentration")
    plt.subplot(133)
    plt.plot(years, e_hist * 2.13)
    plt.xlabel("Years")
    plt.title("Emission in Gigaton")
    plt.ylim(-0.1)
    # plt.savefig(f"./figure/Econ_Climate/T_C_E_{cearth}_{tauc}.pdf")
    Fig_Dir = "./figure/"+args.filename+"/"
    
    os.makedirs(Fig_Dir, exist_ok=True)

    plt.savefig(Fig_Dir+f"T_C_E_{cearth}_{tauc}_{args.maxiter}_{args.fraction}_{args.epsilon}_{args.Xminarr}_{args.Xmaxarr}_{args.hXarr}.pdf")

    

cearth = args.cearth
tauc=args.tauc

T_grid,C_grid,F_grid,Ca,cearth,tauc = model(cearth,tauc)
print("Solving Model Done, cearth={:.4f}, tauc={:.5f}".format(cearth,tauc))
simulation(T_grid,C_grid,F_grid,Ca,cearth,tauc)