# -*- coding: utf-8 -*-
"""
Created on Thu May  2 22:03:10 2024

@author: je.velasco
"""
from myfirstshallowwatermodel.model1d import windstress_1D as sw1d
import numpy as np
import matplotlib.pyplot as plt

model = sw1d(X = 1000, boundary="c", nt=15000, H_0=500, DX=1000,
             plotting=True, plot_interval=50, use_asselin=False,
             wind_duration=1000, wind_speed=-0.05, plot_path='winddriven',
             save_outputs=False, save_interval=100)


H = 500 * np.ones(1000)
X = np.arange(1000)
z = np.linspace(np.pi/2, -np.pi/2, 50)[::-1]
dd = 500 +(2000 * (1 + np.sin(z)) / 2)
H[200:250] = dd
H[250:] = 2500

fig, ax = plt.subplots(1, 1, figsize=(6,4))
fig.suptitle('Perfil batimétrico', fontsize=16)
ax.plot(X, H)
ax.set_xlabel('Longitud X $[km]$')
ax.set_ylabel('Profundidad H $[m]$')
#ax.set_ylim([0, 2200])
#ax.set_xlim([0, 800])
ax.invert_yaxis()
ax.grid()
fig.show()

model.H = H.copy()
model.v2 = model.H * model.DT / model.DX
model.v2 = model.H * model.DT / model.DX
model.v2_ = 2 * model.v2

model.run()

