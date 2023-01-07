# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 12:58:49 2017

Modeling of homopolymer using the method of moments


All units are in mol, L, s, K

@author: Daniel
"""
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# initial concentrations
I_0 = 0.00005  # initial initiator conc. (mol L-1)
M_0 = 0.8 # initial monomer conc. (mol L-1)
S_0 = 55.2  # initial solvent conc. (mol L-1)

# constants
T = 323.15  # (K)
f = 0.5
rxn_time = 10 * 3600 # reaction time in seconds

# rate constants
kd = 9.0 * 10 ** -6  # (s-1) V-50
kp = 254.0  # (L mol-1 s-1)
kt = 30.0 * 10 ** 6  # (L mol-1 s-1)
ktrM = 2.33 ** -2  # (L mol-1 s-1) Odian p. 243
ktrS = 7.34 * 10 ** -3  # (L mol-1 s-1)


# Calculate conversion as a function of time, t (seconds)
#def conversion(t):
#    p = 1 - np.exp(2.0 * (kp + ktrM)
#                   * (2 * f * I_0 / (kd * kt)) ** 0.5
#                   * (np.exp(-kd * t / 2.0) - 1.0))
#    return p

def conversion(t):
    p = 1 - np.exp(-(kp + ktrM) * ((2 * f * kd * I_0 / kt) ** 0.5) * t)
    return p

def moment_equations(variables, t):
    M, I, Y0, Y1, Y2, X0, X1, X2 = variables
    
    Ri = 2 * f * kd * I
    
    ktr = ktrM * M + ktrS * S_0 
    
    DDt = [
           -kp * M * Y0 - ktrM * M * Y0, #dM/dt monomer consumption
           -kd * I, #dI/dt initiator consumption
           Ri - kt * Y0 ** 2, #dY0/dt
           Ri + kp * M * Y0 - ktr * Y1 - kt * Y0 * Y1, #dY1/dt
           Ri + 2 * kp * M * Y1 - ktr * Y2 - kt * Y0 * Y2, #dY2/dt
           ktr * Y0 + kt * Y0 ** 2, #dX0/dt
           kp * M * Y0, #dX1/dt
           ktr * Y2 + kt * Y0 * Y2 #dX2/dt
            ]
    return DDt

time = np.linspace(0, rxn_time, 289)
init_conds = [M_0, I_0, 0, 0, 0, 0, 0, 0]
soln_vec = integrate.odeint(moment_equations, init_conds, time)


NnInst = soln_vec[:,3] / soln_vec[:,2]
NnInst[0] = 0
NwInst = soln_vec[:,4] / soln_vec[:,3]
NwInst[0] = 0
PDIInst = NwInst / NnInst

NnInstDead = soln_vec[:,6] / soln_vec[:,5]
NnInstDead[0] = 0
NwInstDead = soln_vec[:,7] / soln_vec[:,6]
NwInstDead[0] = 0
PDIInstDead = NwInstDead / NnInstDead

RateofReaction = kp * soln_vec[:,0] * soln_vec[:,2]
                                        
CumNn = integrate.cumtrapz(NnInstDead * RateofReaction, time,
                           initial=0) / (integrate.cumtrapz(RateofReaction,
                                           time, initial = 0))
CumNw = integrate.cumtrapz(NwInstDead * RateofReaction, time,
                           initial=0) / (integrate.cumtrapz(RateofReaction,
                                         time, initial = 0))
print("Nn = ", CumNn[-1])
print("Nw = ", CumNw[-1])
print("PDI = ", CumNw[-1]/CumNn[-1])

print("p(t = 18 h) = ", conversion(rxn_time))

plt.figure(1)
plt.xlim(0, 36000)
plt.ylim(0, 8e-8)
plt.xlabel(r'time (s)')
plt.ylabel(r'$Y_0$')
plt.plot(time, soln_vec[:,2])

plt.figure(2)
plt.xlim(0, 36000)
plt.ylim(0, 0.00003)
plt.xlabel(r'time (s)')
plt.ylabel(r'$Y_1$')
plt.plot(time, soln_vec[:,3])

plt.figure(3)
plt.xlim(0, 36000)
plt.ylim(0, 0.03)
plt.xlabel(r'time (s)')
plt.ylabel(r'$Y_2$')
plt.plot(time, soln_vec[:,4])

plt.figure(4)
plt.xlim(0, 36000)
plt.ylim(0, 0.006)
plt.xlabel(r'time (s)')
plt.ylabel(r'$X_0$')
plt.plot(time, soln_vec[:,5])

plt.figure(5)
plt.xlim(0, 36000)
plt.ylim(0, 1.6)
plt.xlabel(r'time (s)')
plt.ylabel(r'$X_1$')
plt.plot(time, soln_vec[:,6])

plt.figure(6)
plt.xlim(0, 36000)
plt.ylim(0, 1000)
plt.xlabel(r'time (s)')
plt.ylabel(r'$X_2$')
plt.plot(time, soln_vec[:,7])

plt.figure(7)
plt.xlim(0, 36000)
plt.ylim(0, 1000)
plt.xlabel(r'time (s)')
plt.ylabel(r'Instantaneous Living Degree of Polymerization')
plt.plot(time, NnInst, label=r'$N_n$')
plt.plot(time, NwInst, label=r'$N_w$')
plt.legend()

plt.figure(8)
plt.xlim(0, 36000)
plt.ylim(0, 1000)
plt.xlabel(r'time (s)')
plt.ylabel(r'Instantaneous Dead Degree of Polymerization')
plt.plot(time, NnInstDead, label=r'$N_n$')
plt.plot(time, NwInstDead, label=r'$N_w$')
plt.legend()

plt.figure(9)
plt.xlim(0, 36000)
plt.ylim(0, 3)
plt.xlabel(r'time (s)')
plt.ylabel(r'Instantaneous PDI')
plt.plot(time, PDIInst, label='living chains')
plt.plot(time, PDIInstDead, label='dead chains')
plt.legend()

plt.figure(10)
plt.xlim(0, 36000)
plt.ylim(0, 1000)
plt.xlabel(r'time (s)')
plt.ylabel(r'Cumulative Degree of Polymerization')
plt.plot(time, CumNn, label='$N_n$')
plt.plot(time, CumNw, label='$N_w$')
plt.legend()

plt.show()
