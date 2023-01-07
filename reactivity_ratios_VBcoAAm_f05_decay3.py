#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:42:32 2019

@author: dvdixon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, Data, RealData
from scipy.integrate import odeint
from scipy.optimize import curve_fit

VBf05_1_df = pd.read_csv('VBcoAAM_r1r2_decay_1.csv')
VBf05_2_df = pd.read_csv('VBcoAAM_r1r2_decay_2.csv')
VBf03_1_df = pd.read_csv('VBcoAAM_r1r2_decay_3.csv')

x_data_VBf05_1 = VBf05_1_df['x_VBf0.5_1']
f1_data_VBf05_1 = VBf05_1_df['f_VBf0.5_1']
F1_data_VBf05_1 = VBf05_1_df['F_VBf0.5_1']

x_data_VBf05_2 = VBf05_2_df['x_VBf0.5_2']
f1_data_VBf05_2 = VBf05_2_df['f_VBf0.5_2']
F1_data_VBf05_2 = VBf05_2_df['F_VBf0.5_2']

x_data_VBf03_1 = VBf03_1_df['x_VBf0.3_1']
f1_data_VBf03_1 = VBf03_1_df['f_VBf0.3_1']
F1_data_VBf03_1 = VBf03_1_df['F_VBf0.3_1']


def mayo_lewis(f1, r1, r2):
    f2 = 1 - f1
    return (r1*f1**2 + f1*f2)/(r1*f1**2 + 2*f1*f2 + r2*f2**2)


def fitfunc(x,r1,r2):
    
    def df1dx(f1,x):
        F = mayo_lewis(f1,r1,r2)
        return (f1 - F)/(1-x)
    
    f1_soln = odeint(df1dx,f10,x)

    return f1_soln[:,0]


def odrfitfunc(r, x):
    
    def df1dx(f1,x):
        F = mayo_lewis(f1,r[0],r[1])
        return (f1 - F)/(1-x)
    
    f1_soln = odeint(df1dx,f10,x)

    return f1_soln[:,0]

f10 = f1_data_VBf03_1[0]

r_fit, r_cov = curve_fit(fitfunc,x_data_VBf03_1,f1_data_VBf03_1,p0=[1.0,0.2],bounds=(0.0,10),method='trf',xtol=1e-6)
print(r_fit)


rxn_data_1 = Data(x_data_VBf03_1,f1_data_VBf03_1)
model = Model(odrfitfunc)
odr = ODR(rxn_data_1, model, [1.0,0.2],maxit=100)
#odr.set_iprint(init=0,iter=0,final=2)
output = odr.run()
output.pprint()

xfit = np.linspace(0,1)
#fit = fitfunc(xfit,r_fit[0],r_fit[1])
fit = fitfunc(xfit,1.057,0.238)
odrfit = odrfitfunc(output.beta, xfit)


plt.plot(x_data_VBf03_1,f1_data_VBf03_1,'o')
plt.plot(xfit,fit,'-')
plt.plot(xfit,odrfit,'--')

plt.ylim(0,1)
