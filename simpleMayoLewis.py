#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:07:26 2019

@author: dvdixon
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

r = np.array([2.479,0.462])
f1dat = np.array([0.097,0.29,0.4, 0.5, 0.5,0.47,0.48,0.64])
F1dat = np.array([0.14,0.49,0.68,0.74,0.76,0.64,0.63,0.79])
#f1dat = [0.098,0.29,0.48,0.49,0.64]
#F1dat = [0.30,0.68,0.95,0.98,0.88]
labf1 = np.array([0.2,0.2,0.4,0.5, 0.5, 0.5])
labF1 = np.array([0.37,0.38,0.68,0.68,0.74,0.76])


def mayo_lewis(r):
    f1 = 0.5
    f2 = 1 - f1
    return (r[0]*f1**2 + f1*f2)/(r[0]*f1**2 + 2*f1*f2 + r[1]*f2**2)

def mayo_lewis_2(ff1, r1, r2):
    f2 = 1 - ff1
    return (r1*ff1**2 + ff1*f2)/(r1*ff1**2 + 2*ff1*f2 + r2*f2**2)


def fitfunc(r):
    F1 =0.64
    return F1 - mayo_lewis(r)


fit = least_squares(fitfunc, r,bounds=(0.0,np.inf))

r_fit, r_cov = curve_fit(mayo_lewis_2,f1dat,F1dat,bounds=(0.0,100))

xfit = np.linspace(0,1)
valfit = mayo_lewis_2(xfit,r_fit[0],r_fit[1])
valfit_2 =mayo_lewis_2(xfit,2.48,0.46)
#valfit_3 =mayo_lewis_2(xfit,5.0,0.27)

plt.plot(1-f1dat,1-F1dat,'o',label='NMR kinetic experiments')
#plt.plot(xfit,valfit,'-')
plt.plot(1-xfit,1-valfit_2,'k-')
#plt.plot(xfit,valfit_3,'b-')
plt.plot([0,1],[0,1],'k--')
plt.plot(1-labf1,1-labF1,'ro',label='low conversion experiments')
plt.legend()
plt.ylim(0,1)
plt.xlim(0,1)
plt.xlabel(r'$\mathrm{f_{Am}}$', fontsize='x-large')
plt.ylabel(r'$\mathrm{F_{Am}}$', fontsize='x-large')
plt.savefig('react_ratio2.png')