#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:33:30 2019

@author: dvdixon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, Data, RealData
from scipy.integrate import odeint
from scipy.optimize import curve_fit

reactiondata_df = pd.read_csv('VBTMAC_conversion.csv')

x_data = reactiondata_df['x_1']
t_data = reactiondata_df['t_1']

def conv_func(r, t):

    I = 0.001
    kd = 8.2 * 10**-6
    f = 0.8
    
    return 1 - np.exp(-r*np.sqrt(f*kd*I)*t)

rxn_data = Data(t_data,x_data)
model = Model(conv_func)

odr = ODR(rxn_data, model, [2.0])
output = odr.run()

output.pprint()

tfit = np.linspace(0,4800)
odrfit = conv_func(output.beta, tfit)

plt.plot(t_data,x_data,'o')
plt.plot(tfit,odrfit,'-')