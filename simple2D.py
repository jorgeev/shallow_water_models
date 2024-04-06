#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:48:55 2024
@author: jevz
"""

#from myfirstshallowwatermodel.model2d import simple2dmodel as sw2d
from myfirstshallowwatermodel.model2d import channel2dmodel as sw2d
import matplotlib.pyplot as plt

model2 = sw2d(X=200, Y=400, H_0=2500, nt=5010, nesting=False, DT=1,omega=10*7.29E-5,
              DX=1000, DY=1000, nestpos=(50,100,30,50), plotting=True, use_asselin=True,
              plot_interval=500, maxh0=3.)

model2.run(cmap='twilight_shifted')

plt.imshow(model2.h1)
plt.show()

plt.contourf(model2.h2)
plt.show()