#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:48:55 2024
Last update Mon Apr 8 10:00  2024
@author: je.velasco@atmosfera.unam.mx
"""

from myfirstshallowwatermodel.model2d import simple2dmodel as sw2d
#from myfirstshallowwatermodel.model2d import channel2dmodel as sw2d
import matplotlib.pyplot as plt

model2 = sw2d(X=300, Y=200, H_0=1500, nt=2000, nesting=False, DT=1, 
              omega=7.29E-5, origin=(150,100), initialc='e',
              DX=1000, DY=1000, nestpos=(50,100,30,50), plotting=True, # use_asselin=True,
              plot_interval=100, size=(4,4), maxh0=1.)

model2.run(cmap='PRGn')#cmap='twilight_shifted')

# fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,6))
# pc1 = ax1.pcolormesh(model2.h2, vmin=-0.5, vmax=0.5, cmap='PRGn')
# plt.colorbar(pc1, ax=ax1,label='h (m)')
# ax1.set_ylabel('y')
# ax1.axhline(100,linestyle='--', color='0.5')
# ax1.set_aspect(1)

# ax2.plot(model2.h2[150,:], 'g-')
# ax2.set_ylabel('h (m)')
# ax2.set_ylim(-0.5,0.5)

# pc3 = ax3.pcolormesh(model2.u2, cmap='RdBu')#, vmin=-1E-3, vmax=1E-3,)
# plt.colorbar(pc3, ax=ax3,label='u (m/s)')
# ax3.set_xlabel('x')
# ax3.axhline(100,linestyle='--', color='0.5')
# ax3.set_aspect(1)

# pc4 = ax4.pcolormesh(model2.v2, cmap='RdBu')#, vmin=-1E-3, vmax=1E-3,)
# plt.colorbar(pc4, ax=ax4,label='v (m/s)')
# ax4.set_xlabel('x')
# ax4.set_ylabel('y')
# ax4.axhline(100,linestyle='--', color='0.5')
# ax4.set_aspect(1)
 
# plt.show()
