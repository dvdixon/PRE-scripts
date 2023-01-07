#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:42:32 2019

@author: dvdixon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.odr import ODR, Model, Data, RealData
from scipy.integrate import odeint
from scipy.optimize import curve_fit

VBf05_1_df = np.genfromtxt('VBcoAAM_r1r2_decay_1.csv', delimiter=',', skip_header=1)
VBf05_2_df = np.genfromtxt('VBcoAAM_r1r2_decay_2.csv', delimiter=',', skip_header=1)
VBf03_1_df = np.genfromtxt('VBcoAAM_r1r2_decay_3.csv', delimiter=',', skip_header=1)
VBf01_1_df = np.genfromtxt('VBcoAAM_r1r2_decay_4.csv', delimiter=',', skip_header=1)
VBf07_1_df = np.genfromtxt('VBcoAAM_r1r2_decay_5.csv', delimiter=',', skip_header=1)

#First column of csv is conversion, then f_VB, then F_VB
x_data_VBf05_1 = VBf05_1_df[:,0]
f1_data_VBf05_1 = VBf05_1_df[:,1]
F1_data_VBf05_1 = VBf05_1_df[:,2]

x_data_VBf05_2 = VBf05_2_df[:,0]
f1_data_VBf05_2 = VBf05_2_df[:,1]
F1_data_VBf05_2 = VBf05_2_df[:,2]

x_data_VBf03_1 = VBf03_1_df[:,0]
f1_data_VBf03_1 = VBf03_1_df[:,1]
F1_data_VBf03_1 = VBf03_1_df[:,2]

x_data_VBf01_1 = VBf01_1_df[:,0]
f1_data_VBf01_1 = VBf01_1_df[:,1]
F1_data_VBf01_1 = VBf01_1_df[:,2]

x_data_VBf07_1 = VBf07_1_df[:,0]
f1_data_VBf07_1 = VBf07_1_df[:,1]
F1_data_VBf07_1 = VBf07_1_df[:,2]

x_data = np.concatenate((x_data_VBf03_1, x_data_VBf01_1))
f1_data = np.concatenate((f1_data_VBf03_1, f1_data_VBf01_1))

lf3 = len(f1_data_VBf03_1)
lf4 = len(f1_data_VBf01_1)

f1_3 = f1_data[:lf3]
f1_4 = f1_data[lf3:]

def mayo_lewis(f1, r1, r2):
    f2 = 1 - f1
    return (r1*f1**2 + f1*f2)/(r1*f1**2 + 2*f1*f2 + r2*f2**2)

    
def combofitfunc(x,r1,r2):
    

    l3 = len(x_data_VBf03_1)
    l4 = len(x_data_VBf01_1)

    x3 = x[:l3]
    x4 = x[l3:]

    
#    x1int = interpolate.interp1d(x1,f1_1)
#    x2int = interpolate.interp1d(x2,f1_2)
#    x3int = interpolate.interp1d(x3,f1_3)
        
    def df1dx(f1,x):
        F = mayo_lewis(f1,r1,r2)
        return (f1 - F)/(1-x)
    
    f3_soln = odeint(df1dx,f1_3[0],x3)
    f4_soln = odeint(df1dx,f1_4[0],x4)

    return np.concatenate((f3_soln[:,0],f4_soln[:,0]))

def comboodrfitfunc(r,x):
    
    x1 = x[:len(x_data_VBf05_1)]
    x2 = x[len(x_data_VBf05_1):(len(x_data_VBf05_2)+len(x_data_VBf05_1))]
    x3 = x[(len(x_data_VBf05_2)+len(x_data_VBf05_1)):]
    
    def df1dx(f1,x):
        F = mayo_lewis(f1,r[0],r[1])
        return (f1 - F)/(1-x)
        
    f1_soln = odeint(df1dx,f1_1[0],x1)
    f2_soln = odeint(df1dx,f1_2[0],x2)
    f3_soln = odeint(df1dx,f1_3[0],x3)
        
    return np.concatenate((f1_soln[:,0],f2_soln[:,0],f3_soln[:,0]))


r_fit, r_cov = curve_fit(combofitfunc,x_data,f1_data,p0=[1.6,0.2],bounds=(0.0,np.inf),method='trf')
print(r_fit)


def fittedfunc(x,f10,r1,r2):
    
    def df1dx(f1,x):
        F = mayo_lewis(f1,r1,r2)
        return (f1 - F)/(1-x)
    
    f1_soln = odeint(df1dx,f10,x)
    
    return f1_soln[:,0]

#rxn_data = Data(x_data,f1_data)
#model = Model(comboodrfitfunc)
#odr = ODR(rxn_data, model, [1.6,0.2])
#odr.set_iprint(init=0,iter=0,final=2)
#output = odr.run()
#output.pprint()

xfit = np.linspace(0,1)
#fit1 = fittedfunc(xfit,f1_1[0],r_fit[0],r_fit[1])
#fit2 = fittedfunc(xfit,f1_2[0],r_fit[0],r_fit[1])
fit3 = fittedfunc(xfit,f1_3[0],r_fit[0],r_fit[1])
fit4 = fittedfunc(xfit,f1_4[0],r_fit[0],r_fit[1])
#fit5 = fittedfunc(xfit,f1_5[0],r_fit[0],r_fit[1])
#odrfit1 = fittedfunc(xfit,f1_1[0],output.beta[0],output.beta[1])
#odrfit2 = fittedfunc(xfit,f1_2[0],output.beta[0],output.beta[1])
#odrfit3 = fittedfunc(xfit,f1_3[0],output.beta[0],output.beta[1])

#plt.plot(x_data_VBf05_1,f1_data_VBf05_1,'o')
#plt.plot(x_data_VBf05_2,f1_data_VBf05_2,'o')
plt.plot(x_data_VBf03_1,f1_data_VBf03_1,'o')
plt.plot(x_data_VBf01_1,f1_data_VBf01_1,'o')
#plt.plot(x_data_VBf07_1,f1_data_VBf07_1,'o')
#plt.plot(xfit,fit1,'-')
#plt.plot(xfit,fit2,'-')
plt.plot(xfit,fit3,'-')
plt.plot(xfit,fit4,'-')
#plt.plot(xfit,fit5,'-')
#plt.plot(xfit,odrfit1,'--')
#plt.plot(xfit,odrfit2,'--')
#plt.plot(xfit,odrfit3,'--')

plt.ylim(0,1)
