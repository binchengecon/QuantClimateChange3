import sys
sys.path.append("../src/")
import numpy as np
import pandas as pd
import pickle
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
from scipy.interpolate import RegularGridInterpolator
import SolveLinSys
from supportfunctions import finiteDiff
import argparse
import os

rcParams["figure.figsize"] = (8,5)
rcParams["savefig.bbox"] = 'tight'

parser = argparse.ArgumentParser(description="values")

parser.add_argument("--maxiter",type=int,default=5000)
parser.add_argument("--simutime",type=float,default=600)

args = parser.parse_args()

# def PDESolver(stateSpace, A, B1, B2, B3, C1, C2, C3, D, v0, 
#               ε = 1, tol = -10):                                              
                                                                                 

#     A = A.reshape(-1,1,order = 'F')                                         
#     B = np.hstack([B1.reshape(-1,1,order = 'F'),B2.reshape(-1,1,order = 'F'),B3.reshape(-1,1,order='F')])
#     C = np.hstack([C1.reshape(-1,1,order = 'F'),C2.reshape(-1,1,order = 'F'),C3.reshape(-1,1,order='F')])
#     D = D.reshape(-1,1,order = 'F')                                         
#     v0 = v0.reshape(-1,1,order = 'F')                                       
#     out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0, ε, tol)           

#     return out                                           


# # Pre-industrial: 282.87K

sa = 1
Ts = 282.9
Cs = 275.5

Q0 = 342.5
p = 0.3
# outgoing radiation linearized
kappa = 1.74
Tkappa = 154
## CO2 radiative forcing
# Greenhouse effect parameter
B = 5.35

alphaland = 0.28
bP = 0.05
bB = 0.08
cod = 3.035
# cearth = 10. # 35 #0.107
# tauc = 20.

cearth = 35.
tauc   = 6603.

coc0 =350
## Ocean albedo parameters
Talphaocean_low = 219
Talphaocean_high = 299
alphaocean_max = 0.84
alphaocean_min = 0.255

Cbio_low = 50
Cbio_high = 700

T0 = 298
C0 = 280

## CO2 uptake by vegetation
wa = 0.015
vegcover = 0.4

Thigh = 315
Tlow = 282
Topt1 = 295
Topt2 = 310
acc = 5

## Volcanism
Volcan = 0.028


# # def alphaocean(T):
# #     """T, matrix, (nT, nC, nF)"""
# #     temp = np.zeros(T.shape)
# #     temp[ T< Talphaocean_low ] = alphaocean_max
# #     temp[ (T>= Talphaocean_low)&(T< Talphaocean_high)] = alphaocean_max + (alphaocean_min - alphaocean_max) / (Talphaocean_high - Talphaocean_low) * (T[(T>= Talphaocean_low)&(T< Talphaocean_high)] - Talphaocean_low)
# #     temp[T>= Talphaocean_high] = alphaocean_min

# #     return temp

# def alphaocean(T):
#     """T, matrix, (nT, nC, nF)"""
#     temp = 0.3444045881126172*np.ones(T.shape)

#     return temp

# #Fraction of ocean covered by ice
# def fracseaice(T):
    
#     temp = np.zeros(T.shape)
#     temp[ T< Talphaocean_low ] = 1
#     temp[ (T>= Talphaocean_low)&(T< Talphaocean_high)] = 1 - 1 / (Talphaocean_high - Talphaocean_low) * (T[(T>= Talphaocean_low)&(T< Talphaocean_high)] - Talphaocean_low)
#     temp[T>= Talphaocean_high] = 0

#     return temp
    


# def biopump(F):
#     """F, accumulated anthrpogenic emission"""
#     temp = np.zeros(F.shape)
    
#     temp[F < Cbio_low] = 1
#     temp[(F >= Cbio_low)&(F < Cbio_high)] = 1 - 1/(Cbio_high - Cbio_low) * (F[(F >= Cbio_low)&(F < Cbio_high)] - Cbio_low)
#     temp[F >= Cbio_high] = 0
#     return temp


# def veggrowth(T):
    
#     temp = np.zeros(T.shape)
    
#     temp[T < Tlow] = 0
#     temp[(T >= Tlow)&(T < Topt1)] = acc / (Topt1 - Tlow) * (T[(T >= Tlow)&(T < Topt1)] - Tlow)
#     temp[(T >= Topt1)&(T < Topt2)] = acc
#     temp[(T >= Topt2)&(T < Thigh)] = acc / (Topt2 - Thigh) * (T[(T >= Topt2)&(T < Thigh)] - Thigh)
#     temp[T > Thigh] = 0
    
#     return temp


# #Incoming radiation modified by albedo
# def Ri(T):
#     return 1/cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean(T)))

# # Outgoing radiation modified by greenhouse effect
# def Ro(T, C):
#     return 1/cearth * (kappa * (T - Tkappa) -  B * np.log(C / C0))

# #Solubility of atmospheric carbon into the oceans
# # carbon pumps
# def kappaP(T):
#     return np.exp(-bP * (T - T0))

# def oceanatmphysflux(T):
#     return 1 / tauc * (coc0 * (np.exp(-bP * (T - T0))))

# def oceanbioflux(T, F, sa):
    
#     if sa == 1:
        
#         return 1/tauc * (coc0 * (np.exp(bB * biopump(F) * (T - T0))))
    
#     elif sa == 0:
        
#         return 1/tauc * (coc0 * (np.exp(bB * (T - T0))))
    
#     else:
#         return ValueError("Wrong input value: 0 or 1.")

# def oceanatmcorrflux(C):
#     return 1 / tauc * (- cod * C)


# # # Economic paramaters
# # gamma_1 = 1.7675 / 10000.
# # gamma_2 = 2 * 0.0022
# # delta   = 0.01
# # eta     = 0.032

# # # State variable
# # # Temperature anomaly, in celsius
# # T_min  = 1e-8 
# # T_max  = 10. # 
# # hT     = 0.2
# # T_grid = np.arange(T_min, T_max + hT, hT)

# # # atmospheric carbon concentration, in ppm
# # C_min  = 200
# # C_max  = 400.
# # hC     = 4.
# # C_grid = np.arange(C_min, C_max + hC, hC)

# # # F, Sa in the notes, accumulative anthropogenic carbon, in gigaton, since 1800
# # F_min = 1e-8 # 10. avoid 
# # F_max = 2000. # 2500 x2.13 gm # # on hold -> 4000 / 2.13 ppm
# # hF = 40.
# # F_grid = np.arange(F_min, F_max + hF, hF)

# # # meshgrid
# # (T_mat, C_mat, F_mat) = np.meshgrid(T_grid, C_grid, F_grid, indexing="ij")
# # stateSpace = np.hstack([
# #     T_mat.reshape(-1, 1, order="F"),
# #     C_mat.reshape(-1, 1, order="F"),
# #     F_mat.reshape(-1, 1, order="F")
# # ])

# # T_mat.shape

# # Economic paramaters
gamma_1 = 1.7675 / 10000.
gamma_2 = 2 * 0.0022
delta   = 0.01
eta     = 0.032

# State variable
# Temperature anomaly, in celsius
T_min  = 1e-8 
T_max  = 10. # 
hT     = 0.2
T_grid = np.arange(T_min, T_max + hT, hT)

# atmospheric carbon concentration, in gigaton
C_min  = 200.
C_max  = 400.
hC     = 10.
C_grid = np.arange(C_min, C_max + hC, hC)

# F, Sa in the notes, accumulative anthropogenic carbon, in gigaton, since 1800
F_min = 0. # 10. avaoid 
F_max = 300. # 2500 x2.13 gm # # on hold -> 4000 / 2.13 ppm
hF = 10.
F_grid = np.arange(F_min, F_max + hF, hF)

# # meshgrid
# (T_mat, C_mat, F_mat) = np.meshgrid(T_grid, C_grid, F_grid, indexing="ij")
# stateSpace = np.hstack([
#     T_mat.reshape(-1, 1, order="F"),
#     C_mat.reshape(-1, 1, order="F"),
#     F_mat.reshape(-1, 1, order="F")
# ])

# T_mat.shape





To = 282.87 # Mean with no anthropogenic carbon emissions, in Fᵒ

# cearth = 35.
# tauc   = 6603.

# # v0 = pickle.load(open("data_35.0_6603", "rb"))["v0"]
# v0 =  - eta * T_mat - eta * F_mat
# # v0 =  delta * eta * np.log(delta /4 * (9000/2.13 - F_mat)) + (eta - 1) * gamma_2 * T_mat / cearth * (B * np.log(C_mat/ C0) + kappa * (T_mat + To - Tkappa))

# dG  = gamma_1 + gamma_2 * T_mat
# epsilon  = 0.05
# count    = 0
# error    = 1.
# tol      = 1e-8
# max_iter = args.maxiter
# fraction = 0.05


# while error > tol and count < max_iter:
    
#     dvdT  = finiteDiff(v0, 0, 1, hT)
#     dvdTT = finiteDiff(v0, 0, 2, hT)
#     dvdC  = finiteDiff(v0, 1, 1, hC)
# #     dvdC[dvdC >= - 1e-16] = - 1e-16
#     dvdCC = finiteDiff(v0, 1, 2, hC)
#     dvdF  = finiteDiff(v0, 2, 1, hF)
#     dvdFF = finiteDiff(v0, 2, 2, hF)
        

#     Ca = - eta * delta / (dvdC + dvdF)

#     Ca[Ca <= 1e-32] = 1e-32
    
#     if count >=1:
#         Ca = Ca * fraction + Ca_star * (1 - fraction)
    
# #     Ca = np.ones(T_mat.shape)
#     A  = - delta * np.ones(T_mat.shape)
#     B1 = Ri(T_mat + To) - Ro(T_mat + To, C_mat)
#     B2 = Volcan
#     B2 += Ca * sa
#     B2 -= wa * C_mat * vegcover * veggrowth(T_mat +To)
#     B2 += oceanatmphysflux(T_mat + To)  * (1 - fracseaice(T_mat + To))
#     B2 += oceanbioflux(T_mat + To, F_mat, sa) * (1 - fracseaice(T_mat + To))
#     B2 += oceanatmcorrflux(C_mat) * (1 - fracseaice(T_mat + To))
#     B3 = Ca
#     C1 = 0.0 * np.ones(T_mat.shape)
#     C2 = 0.0 * np.ones(T_mat.shape)
#     C3 = np.zeros(T_mat.shape)
#     D  = eta * delta * np.log(Ca) + (eta - 1) * dG * B1

#     out = PDESolver(stateSpace, A, B1, B2, B3, C1, C2, C3, D, v0, epsilon)
#     v = out[2].reshape(v0.shape, order="F")

#     rhs_error = A * v0 + B1 * dvdT + B2 * dvdC + B3 * dvdF + C1 * dvdTT + C2 * dvdCC + C3 * dvdFF + D
#     rhs_error = np.max(abs(rhs_error))
#     lhs_error = np.max(abs((v - v0)/epsilon))

#     error = lhs_error
#     v0 = v
#     Ca_star = Ca
#     count += 1

#     # print("Iteration: %s;\t False Transient Error: %s;\t PDE Error: %s\t" % (count, lhs_error, rhs_error),flush=True)

# print("Total iteration: %s;\t LHS Error: %s;\t RHS Error %s\t" % (count, lhs_error, rhs_error),flush=True)


# res = [v0,T_grid,C_grid,F_grid,Ca]

Output_Dir = "/scratch/bincheng/"
Data_Dir = Output_Dir+"QuantClimateChange/data/"

# os.makedirs(Data_Dir, exist_ok=True)


# with open(Data_Dir +"model_Tmin_{:.2f}_Tmax_{:.2f}_Cmin_{:.2f}_Cmax_{:.2f}_Fmin_{:.2f}_Fmax_{:.2f}".format(T_min,T_max,C_min,C_max,F_min,F_max), "wb") as f:
#     pickle.dump(res,f)



with open(Data_Dir +"model_Tmin_{:.2f}_Tmax_{:.2f}_Cmin_{:.2f}_Cmax_{:.2f}_Fmin_{:.2f}_Fmax_{:.2f}".format(T_min,T_max,C_min,C_max,F_min,F_max), "rb") as f:
    Guess = pickle.load(f)


T_grid=Guess[1]
C_grid=Guess[2]
F_grid=Guess[3]
Ca=Guess[4]
t_max = args.simutime
# dt = 1/12
dt = 1  # , Gigaton per year
gridpoints = (T_grid, C_grid, F_grid)   
Ca_func = RegularGridInterpolator(gridpoints, Ca)

T_0 = To + min(T_grid)
C_0 = 275.5
# F_0 = min(F_grid) #(870 - 580) / 2.13 # total cumulated, as of now, preindustrial with Fo
F_0 = 0.1 #(870 - 580) / 2.13 # total cumulated, as of now, preindustrial with Fo

# T_0 = To + 1.1
# C_0 = 417
# F_0 = (870 - 580) / 2.13

def get_e(x):
    return Ca_func([x[0] - To, x[1], x[2]])

# Ocean albedo
def alphaocean_1d(T):
    if T < Talphaocean_low:
        return alphaocean_max
    elif T < Talphaocean_high:
        return alphaocean_max + (alphaocean_min - alphaocean_max) / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    else: # so T is higher
        return alphaocean_min

# Vegetation growth function
def veggrowth_1d(T):
    if T < Tlow:
        return 0
    if (T >= Tlow) and (T < Topt1):
        return acc / (Topt1 - Tlow) * (T - Tlow)
    if (T >= Topt1) and (T <= Topt2):
        return acc
    if (T > Topt2) and (T < Thigh):
        #return acc
        return acc / (Topt2 - Thigh) * (T - Thigh)
    if T > Thigh:
        #return acc
        return 0

def oceanatmphysflux_1d(T):
    return 1 / tauc * (coc0 * (np.exp(-bP * (T - T0))))

def fracseaice_1d(T):
    if T < Talphaocean_low:
        return 1
    elif T < Talphaocean_high:
        return 1 - 1 / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    else: # so T is higher
        return 0

def biopump_1d(Cc):
    if Cc < Cbio_low:
        return 1
    elif Cc < Cbio_high:
        return 1 - 1 / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
    else: 
        return 0


def oceanbioflux_1d(T, F, sa):
     return 1/tauc * (coc0 * (np.exp(bB * biopump_1d(F) * (T - T0))))

def oceanatmcorrflux_1d(C):
    return 1 / tauc * (- cod * C)


def mu_T(x):
    Ri_t = 1 / cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean_1d(x[0])))
    Ro_t = 1 / cearth * (kappa * (x[0] - Tkappa) -  B * np.log(x[1] / C0))
    return Ri_t - Ro_t

def mu_C(x):
    Ca_t = Ca_func([x[0] - To, x[1], x[2]])
    dC = Volcan
    dC += Ca_t * sa
    dC -= wa * x[1] * vegcover * veggrowth_1d(x[0])
    dC += oceanatmphysflux_1d(x[0]) * (1 - fracseaice_1d(x[0]))
    dC += oceanbioflux_1d(x[0], x[2], sa) * (1 - fracseaice_1d(x[0]))
    dC += oceanatmcorrflux_1d(x[1]) * (1 - fracseaice_1d(x[0]))
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

    print("Time={:.2f}, T={:.2f}, C={:.2f}, g={:.2f} \n" .format(tm, hist[tm,0], hist[tm,1], e_hist[tm]))


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
plt.savefig(f"./figure/Econ_Climate/Surialpha_T_C_E_{cearth}_{tauc}_{t_max}.pdf")