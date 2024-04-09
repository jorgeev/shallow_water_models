#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:48:55 2024
Last update Mon Apr 8 22:00  2024
@author: je.velasco
"""

from myfirstshallowwatermodel.model2d import simple2dmodel as sw2d
#from myfirstshallowwatermodel.model2d import channel2dmodel as sw2d
import matplotlib.pyplot as plt

model2 = sw2d(X=300, Y=200, H_0=1500, nt=10000, nesting=False, DT=1, 
              omega=7.29E-5, origin=(150,100), initialc='e', use_asselin=True,
              asselin_value=0.1, asselin_step=1, calculate_metrics=True,
              DX=1000, DY=1000, nestpos=(50,100,30,50), plotting=True, # use_asselin=True,
              plot_interval=1000, size=(4,4), maxh0=1.)

model2.run(cmap='PRGn')#cmap='twilight_shifted')


fig, ax = plt.subplots(4,1,figsize=(12,6), dpi=300, sharex=True)
ax[0].plot(model2.E_k)
ax[1].plot(model2.E_p)
ax[2].plot(model2.E_p + model2.E_k)
ax[3].plot(model2.vol)
plt.show()

fig, ax = plt.subplots(1,1,figsize=(12,6), dpi=300)
ax.plot((model2.E_p + model2.E_k)[100:])
plt.show()
