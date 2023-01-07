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

VBf0.5_r1r2_1_df = pd.read_csv('VBcoAAM_r1r2_1.csv')
VBf0.5_r1r2_2_df = pd.read_csv('VBcoAAM_r1r2_2.csv')
VBf0.3_r1r2_1_df = pd.read_csv('VBcoAAM_r1r2_3.csv')


f1_data = reactiondata_df['f_VB_1']
F1_data =reactiondata_df['F_VB_avg_1']
x_data = reactiondata_df['x_1']

f1_data_2 = reactiondata2_df['f_VB_2']
F1_data_2 =reactiondata2_df['F_VB_avg_2']
x_data_2 = reactiondata2_df['x_2']


def mayo_lewis(f1, r1, r2):
    f2 = 1 - f1
    return (r1*f1**2 + f1*f2)/(r1*f1**2 + 2*f1*f2 + r2*f2**2)

#def df1dx(f1,x):
#    F = mayo_lewis(f1,r1,r2)
#    return (f1 - F)/(1-x)

def fitfunc(x,r1,r2):
    
    def df1dx(f1,x):
        F = mayo_lewis(f1,r1,r2)
        return (f1 - F)/(1-x)
    
    f10 = f1_data[0]
    f1_soln = odeint(df1dx,f10,x)
#    F1 = (f10 - f1_soln[:,0]*(1-x))/x #test
#    F1[0] = f1_data[0]
    return f1_soln[:,0]
#    return F1 #test

def odrfitfunc(r, x):
    
    def df1dx(f1,x):
        F = mayo_lewis(f1,r[0],r[1])
        return (f1 - F)/(1-x)
    
    f10 = f1_data[0]
    f1_soln = odeint(df1dx,f10,x)

    return f1_soln[:,0]


r_fit_1, r_cov_1 = curve_fit(fitfunc,x_data,f1_data,p0=[2.0,0.5],bounds=(0.0,np.inf),method='trf',xtol=1e-6)
r_fit_2, r_cov_2 = curve_fit(fitfunc,x_data_2,f1_data_2,p0=[2.0,0.5],bounds=(0.0,np.inf),method='trf',xtol=1e-6)
#r_fit_1, r_cov_1 = curve_fit(fitfunc,x_data,F1_data,p0=[1.0,0.5],bounds=(0.0,np.inf),method='trf',xtol=1e-6)
#r_fit_2, r_cov_2 = curve_fit(fitfunc,x_data_2,F1_data_2,p0=[1.0,0.5],bounds=(0.0,np.inf),method='trf',xtol=1e-6)
print(r_fit_1)
print(r_fit_2)

rxn_data = Data(x_data_2,f1_data_2)
model = Model(odrfitfunc)
odr = ODR(rxn_data, model, [1.0,0.5])
output = odr.run()
output.pprint()



xfit = np.linspace(0,1)
fit = fitfunc(xfit,r_fit_1[0],r_fit_1[1])
odrfit = odrfitfunc(output.beta, xfit)


plt.plot(x_data,f1_data,'o')
plt.plot(xfit,fit,'-')
plt.plot(xfit,odrfit,'--')
plt.ylim(0,1)
