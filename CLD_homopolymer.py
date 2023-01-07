#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:26:59 2019

Modeling instantaneous chain length distributions for a homopolymer and
integrating that over the reaction time.

@author: dvdixon
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# initial concentrations
I_0 = 0.0005  # initial initiator conc. (mol L-1)
M_0 = 1.0  # initial monomer conc. (mol L-1)
S_0 = 55.2  # initial solvent conc. (mol L-1)

# constants
T = 323.15  # (K)
f = 0.8 # initiator efficiency

rxn_time = 4 * 3600 # reaction time in seconds
time_steps = 360
max_chain_length = 100000

# rate constants
kd = 8.2 * 10 ** -6  # (s-1) V-50
kp = 254.0 * 4 # (L mol-1 s-1)
ktc = 30.0 * 10 ** 6  # (L mol-1 s-1)
ktd = 0 * 10 ** 6  # (L mol-1 s-1)
ktrM = 2.33 * 10 ** -2  # (L mol-1 s-1) Odian p. 243
ktrS = 7.34 * 10 ** -3  # (L mol-1 s-1)

kt = ktc + ktd

def conversion(t):
    p = 1 - np.exp(-(kp + ktrM) * ((2 * f * kd * I_0 / kt) ** 0.5) * t)
    return p

time = np.linspace(0,rxn_time,time_steps)

monomer_conc = M_0 * (1 - conversion(time))

P_star = (2 * f * kd * I_0 / kt) ** 0.5

Rp = kp * monomer_conc * P_star

beta = ktc * P_star / (kp * monomer_conc)
tau = (ktd * P_star / (kp * monomer_conc)) + (ktrM/kp) + (ktrS * S_0 / (kp * monomer_conc))

Nn = 1 / (tau + beta/2)
Nw = (2 * tau + 3 * beta) / (tau + beta) ** 2
PDI = Nw/Nn

CumNn = integrate.cumtrapz(Nn * Rp, time) / (integrate.cumtrapz(Rp, time))
CumNw = integrate.cumtrapz(Nw * Rp, time) / (integrate.cumtrapz(Rp, time))

cld_n = np.zeros((max_chain_length,time_steps))
cld_w = np.zeros((max_chain_length,time_steps))

for i in range(1,max_chain_length):
    cld_n[i,:] = (tau + 0.5 * beta * (beta + tau) * i) * np.exp(-(beta + tau) * i)
    cld_w[i,:] = i * (beta + tau) * (tau + 0.5 * beta * (beta + tau) * i) * np.exp(-(beta + tau) * i)

cumCLD_n = np.zeros((max_chain_length,time_steps-1))
cumCLD_w = np.zeros((max_chain_length,time_steps-1))

for i in range(1,max_chain_length):
    cumCLD_n[i,:] = integrate.cumtrapz(cld_n[i,:] * Rp, time) / integrate.cumtrapz(Rp,time)
    cumCLD_w[i,:] = integrate.cumtrapz(cld_w[i,:] * Rp, time) / integrate.cumtrapz(Rp,time)

chain_lengths = np.arange(max_chain_length)
log_chain_lengths = np.log10(chain_lengths)

plt.figure(1)
plt.xlabel(r'log(i)')
plt.ylabel(r'$W(i)$')
plt.plot(log_chain_lengths, cumCLD_w[:,-1])
plt.show()

print("Nn = ", CumNn[-1])
print("Nw = ", CumNw[-1])
print("PDI = ", CumNw[-1]/CumNn[-1])
print("p(t = 4 h) = ", conversion(rxn_time))