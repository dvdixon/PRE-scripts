#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:26:59 2019

Modeling instantaneous chain length distributions
and chemical composition distributions for a homopolymer and
integrating that over the reaction time.

@author: dvdixon
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy import optimize


# initial concentrations
I_0 = 0.00005  # initial initiator conc. (mol L-1)
M_0 = 1.0  # initial monomer conc. (mol L-1)
S_0 = 55.2  # initial solvent conc. (mol L-1)

# constants
T = 323.15  # (K)
f = 0.5 # initiator efficiency

rxn_time = 10 * 3600 # reaction time in seconds
time_steps = 360
max_chain_length = 100000


# rate constants
kd = 9.0 * 10 ** -6  # (s-1) V-50
kp = 254.0  # (L mol-1 s-1)
ktc = 30.0 * 10 ** 6  # (L mol-1 s-1)
ktd = 0 * 10 ** 6  # (L mol-1 s-1)
ktrM = 2.33 * 10 ** -2  # (L mol-1 s-1) Odian p. 243
ktrS = 7.34 * 10 ** -3  # (L mol-1 s-1)

# copolymer constants
r1 = 5.0
r2 = 0.2
f1_0 = 0.5
f2_0 = 1 - f1_0
alpha = r2 / (1-r2)
beta = r1 / (1-r1)
gamma = (1 - r1*r2)/((1-r1)*(1-r2))
delta = (1-r2)/(2-r1-r2)


kt = ktc + ktd

def conversion(t):
    p = 1 - np.exp(-(kp + ktrM) * ((2 * f * kd * I_0 / kt) ** 0.5) * t)
    return p

time = np.linspace(0,rxn_time,time_steps)

conv = conversion(time)

monomer_conc = M_0 * (1 - conv)

def conv_resid(f_1, M):
    return (M/M_0) - (((f_1/f1_0)**alpha)*(((1-f_1)/f2_0)**beta)*(((f1_0-delta)/(f_1-delta))**gamma))

f1 = f1_0 * np.ones(len(time))

    
for j in range(len(f1)):
    f1[j] = optimize.fsolve(conv_resid,f1_0, args=monomer_conc[j], xtol=0.00001)

P_star = (2 * f * kd * I_0 / kt) ** 0.5

Rp = kp * monomer_conc * P_star

rho = ktc * P_star / ((ktc + ktd) * P_star + ktrS * S_0 + ktrM * monomer_conc)

beta = ktc * P_star / (kp * monomer_conc)
tau = (ktd * P_star / (kp * monomer_conc)) + (ktrM/kp) + (ktrS * S_0 / (kp * monomer_conc))

F_avg = (r1 * f1 ** 2 + f1 * (1-f1)) / (r1 * f1 ** 2 + 2 * f1 * (1-f1) + r2 * (1-f1) ** 2)
b = F_avg*(1-F_avg)*(1 - 4*F_avg*(1-F_avg)*(1-r1*r2))**0.5

y = np.zeros((101,time_steps))
for j in range(0,time_steps):
    for k in range(0,101):
        y[k,j] = k * 0.01 - F_avg[j]

Nn = 1 / (tau + beta/2)
Nw = (2 * tau + 3 * beta) / (tau + beta) ** 2
PDI = Nw/Nn

CumNn = integrate.cumtrapz(Nn * Rp, time) / (integrate.cumtrapz(Rp, time))
CumNw = integrate.cumtrapz(Nw * Rp, time) / (integrate.cumtrapz(Rp, time))

cld_n = np.zeros((max_chain_length,time_steps))
cld_w = np.zeros((max_chain_length,time_steps))
ccd_w = np.zeros((max_chain_length,101,time_steps))

for i in range(1,max_chain_length):
    cld_n[i,:] = (tau + 0.5 * beta * (beta + tau) * i) * np.exp(-(beta + tau) * i)
    cld_w[i,:] = i * (beta + tau) * (tau + 0.5 * beta * (beta + tau) * i) * np.exp(-(beta + tau) * i)
    for f in range(0,101):
        ccd_w[i,f,:] = 2.3026 * i**2 * (tau+beta)**2 * (1.0 - rho + 0.5*rho*i*(tau+beta))*np.exp(-i*(tau+beta))*((i/(2.0*np.pi*b))**0.5)*np.exp(-i*(y[f]**2)/(2.0*b))


cumCLD_n = np.zeros((max_chain_length,time_steps-1))
cumCLD_w = np.zeros((max_chain_length,time_steps-1))
#cumCCD_w = np.zeros((max_chain_length,101,time_steps-1))
cumCCD_w = np.zeros((max_chain_length,101))
#CCD_w = np.zeros((101,time_steps-1))
CCD_w = np.zeros(101)

for i in range(1,max_chain_length):
    cumCLD_n[i,:] = integrate.cumtrapz(cld_n[i,:] * Rp, time) / integrate.cumtrapz(Rp,time)
    cumCLD_w[i,:] = integrate.cumtrapz(cld_w[i,:] * Rp, time) / integrate.cumtrapz(Rp,time)
    for f in range(0,101):
#        cumCCD_w[i,f,:] = integrate.cumtrapz(ccd_w[i,f,:] * Rp, time) / integrate.cumtrapz(Rp,time)
#        CCD_w[f] += cumCCD_w[i,f,:]
        cumCCD_w[i,f] = integrate.trapz(ccd_w[i,f] * Rp, time) / integrate.trapz(Rp,time)
        CCD_w[f] += cumCCD_w[i,f]

CCD_w_sum = 0.0

for f in range(0,101):
    CCD_w_sum += CCD_w[f]

for f in range(0,101):
    CCD_w[f] = CCD_w[f]/CCD_w_sum

chain_lengths = np.arange(max_chain_length)
log_chain_lengths = np.log10(chain_lengths)
mol_frac = np.linspace(0,1,101)

plt.figure(1)
plt.xlabel(r'log(i)')
plt.ylabel(r'$W(i)$')
plt.plot(log_chain_lengths, cumCLD_w[:,-1])

plt.figure(2)
plt.pcolormesh(cumCCD_w)
plt.figure(3)
plt.xlim(0.7,1.0)
plt.plot(mol_frac, CCD_w)
#plt.figure(4)
#plt.matshow(cumCCD_w)
plt.show()
