#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:48:55 2024
@author: jevz
"""

from myfirstshallowwatermodel.model2d import simple2dmodel as sw2d
import matplotlib.pyplot as plt

model2 = sw2d(X=800, Y=200, H_0=2500, nt=5000, nesting=False, DT=3, omega=20*7.29E-5,
              DX=1000, DY=1000, nestpos=(50,100,30,50), plotting=True, 
              plot_interval=200, initialc='D', origin=(100,100), size=(4,4), 
              maxh0=3.)

model2.run(cmap='twilight_shifted')

plt.contourf(model2.h2)