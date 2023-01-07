# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 20:51:07 2017

Simple Monte Carlo Simulation for Polymerization Reactions

Assumes only

Ri
Rp
Rtd
Rtc

No transfer reactions

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# Number of Monte-Carlo Trials
trials = 3000

#constants
Na = 6.022 * 10 ** 23
kp = 705.0 / Na
ktd = 50.0 * 10 ** 6 / Na
ktc = 50.0 * 10 ** 6 / Na

M_0 = 1.9976

f = 0.5
kd = 8.5 * 10 ** -6
I_0 = 0.02

P_dot = ((2 * f * kd * I_0 / ((ktd + ktc) * Na) ) ** 0.5) * Na

def prop_rate(M, Prad):
    return kp * M

def disp_rate(Prad):
    return ktd * Prad

def comb_rate(Prad):
    return ktc * Prad

def total_rate(M, Prad):
    return prop_rate(M, Prad) + disp_rate(Prad) + comb_rate(Prad)

def P_Rp(M, Prad):
    return prop_rate(M, Prad) / total_rate(M, Prad)

def P_Rtd(M, Prad):
    return disp_rate(Prad) / total_rate(M, Prad)

def P_Rtc(M, Prad):
    return comb_rate(M, Prad) / total_rate(M, Prad)

def radical_conc(LivingChains):
    Prad = 0
    for i in range(1, len(LivingChains)):
        Prad = Prad + LivingChains[i]
    return Prad

def random_chain(chain_list):
    rnd_chain = random.randrange(0, len(chain_list))
    return rnd_chain
'''
def random_chain(chain_list):
    total_chains = 0.0
    chain_dist = np.zeros(len(chain_list))
    cum_prob = 0.0
    chain_index = 1
    for i in range(1, len(chain_list)):
        total_chains = total_chains + chain_list[i]
    for j in range(1, len(chain_list)):
        chain_dist[j] = chain_list[j] / total_chains
    rnd_num = random.random()
    for k in range(1, len(chain_list)):
        cum_prob = cum_prob + chain_dist[k]
        if rnd_num <= cum_prob and chain_dist[k] > 0 and chain_index > 1:
            chain_index = k

    return chain_index
'''    
    
NumOfMonomers = M_0 * Na
#MaxChainLength = 1000

LivingChains = [1] * 100
DeadChains = [0]

#LivingChains = np.zeros(MaxChainLength)
#DeadChains = np.zeros(MaxChainLength)

#DeadChains[0] = NumOfMonomers
#LivingChains[1] = P_dot
          
for m in range(trials):

    if len(LivingChains) < 2:
        LivingChains.extend([1,1])
    
    live = True
    rnd_chain = random_chain(LivingChains)

    while live == True:
        
        rnd_rxn = random.random()
        prop_prob = P_Rp(NumOfMonomers, P_dot)
        disp_prob = P_Rtd(NumOfMonomers, P_dot)
        

        if (rnd_rxn <= prop_prob and NumOfMonomers > 0
            and LivingChains[rnd_chain] > 0): #propagation
            
            LivingChains[rnd_chain] = LivingChains[rnd_chain] + 1
            NumOfMonomers = NumOfMonomers - 1
                      
        elif rnd_rxn <= (prop_prob + disp_prob): #disproportiunation
            rnd_chain2 = random_chain(LivingChains)
            if rnd_chain2 == rnd_chain:
                rnd_chain2 = rnd_chain2 - 1
            live2 =True
            while live2 == True:
                rnd_rxn2 = random.random()
                prop_prob = P_Rp(NumOfMonomers, P_dot)
                if (rnd_rxn2 <= prop_prob and NumOfMonomers > 0
                    and LivingChains[rnd_chain] > 0):
                    
                    LivingChains[rnd_chain2] = LivingChains[rnd_chain2] + 1
                    NumOfMonomers = NumOfMonomers - 1
                else:
                    live2 = False      
            if LivingChains[rnd_chain] >= 1 and LivingChains[rnd_chain2] >= 1:
                DeadChains.extend([LivingChains[rnd_chain],
                               LivingChains[rnd_chain2]])
                if rnd_chain > rnd_chain2:
                    LivingChains.pop(rnd_chain)
                    LivingChains.pop(rnd_chain2)
                else:
                    LivingChains.pop(rnd_chain2)
                    LivingChains.pop(rnd_chain) 
                live = False
        else: #combination
            rnd_chain2 = random_chain(LivingChains)
            if rnd_chain2 == rnd_chain:
                rnd_chain2 = rnd_chain2 - 1
            live2 =True
            while live2 == True:
                rnd_rxn2 = random.random()
                prop_prob = P_Rp(NumOfMonomers, P_dot)
                if (rnd_rxn2 <= prop_prob and NumOfMonomers > 0
                    and LivingChains[rnd_chain] > 0):
                    
                    LivingChains[rnd_chain2] = LivingChains[rnd_chain2] + 1
                    NumOfMonomers = NumOfMonomers - 1
                else:
                    live2 = False
            if LivingChains[rnd_chain] >= 1 and LivingChains[rnd_chain2] >= 1:
                DeadChains.append(LivingChains[rnd_chain]
                              + LivingChains[rnd_chain2])
                if rnd_chain > rnd_chain2:
                    LivingChains.pop(rnd_chain)
                    LivingChains.pop(rnd_chain2)
                else:
                    LivingChains.pop(rnd_chain2)
                    LivingChains.pop(rnd_chain)
                live = False

plt.hist(DeadChains, bins=100 , range=[1, max(DeadChains)])

def calc_ni(chain_array):
    ni = 0
    n = np.zeros(max(chain_array) + 1)
    for i in range(1, max(chain_array) + 1):
        for j in range(1, len(chain_array)):
            if chain_array[j] == i:
                n[i] = n[i] + 1
    
    for k in range(1, max(chain_array) + 1):
        ni = ni + n[k]
    return ni

def calc_ini(chain_array):
    ini = 0
    n = np.zeros(max(chain_array) + 1)
    for i in range(1, max(chain_array) + 1):
        for j in range(1, len(chain_array)):
            if chain_array[j] == i:
                n[i] = n[i] + 1
                 
    for k in range(1, max(chain_array) + 1):
        ini = ini + k * n[k]
    return ini

def calc_iini(chain_array):
    iini = 0
    n = np.zeros(max(chain_array) + 1)
    for i in range(1, max(chain_array) + 1):
        for j in range(1, len(chain_array)):
            if chain_array[j] == i:
                n[i] = n[i] + 1
                 
    for k in range(1, max(chain_array) + 1):
        iini = iini + k * k * n[k]
    return iini

Nn = calc_ini(DeadChains) / calc_ni(DeadChains)
Nw = calc_iini(DeadChains) / calc_ini(DeadChains)
PDI = Nw / Nn
print(Nn)
print(Nw)
print(PDI)



