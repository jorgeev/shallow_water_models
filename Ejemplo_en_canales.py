# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:30 2024

@author: je.velasco
"""

from myfirstshallowwatermodel.model2d import channel2dmodel as sw2d
import matplotlib.pyplot as plt

model2 = sw2d(X=200, Y=400, H_0=2500, nt=5000, nesting=True, DT=1, 
              omega=20*7.29E-5, use_asselin=False, period=500,
              asselin_value=0.1, asselin_step=1, calculate_metrics=True,
              DX=1000, DY=1000, nestpos=(140,100,55,200), plotting=True, 
              nest_ratio=5,
              plot_interval=50)

model2.run(cmap='twilight_shifted')

fig, ax = plt.subplots(4,1,figsize=(12,6), dpi=300, sharex=True)
ax[0].plot(model2.E_k)
ax[1].plot(model2.E_p)
ax[2].plot(model2.E_p + model2.E_k)
ax[3].plot(model2.vol)
ax[1].grid()
ax[2].grid()
ax[3].grid()
ax[0].grid()
plt.show()

#fig, ax = plt.subplots(1,1,figsize=(12,6), dpi=300)
#ax.plot((model2.E_p + model2.E_k)[1500:])
#plt.show()
