# -*- coding: utf-8 -*-
"""
Created on Thu May  2 22:03:10 2024

@author: je.velasco
"""
from myfirstshallowwatermodel.model1d import windstress_1D as sw1d
import numpy as np
import matplotlib.pyplot as plt

X = 1000
model = sw1d(X = X, boundary="c", nt=15000, H_0=500, DX=1000,
             plotting=True, plot_interval=50, use_asselin=True, 
             asselin_step=500, asselin_value=0.01, 
             wind_duration=2000, wind_speed=-0.02, plot_path='wind_model',
             save_outputs=False, save_interval=100)


H = 500 * np.ones(X)
X = np.arange(X)
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

fig, ax = plt.subplots(4,1,figsize=(6,8),sharex=True, dpi=300)
fig.suptitle('Métricas', fontsize=16)
ax[0].set_title('Energía cinética')
ax[1].set_title('Energía potencial')
ax[2].set_title('Energía potencial + Energía cinética')
ax[3].set_title('Volumen')

ax[0].plot(model.E_k, label='periódica', linewidth=0.5)

ax[1].plot(model.E_p, linewidth=0.5)

ax[2].plot(model.E_p + model.E_k, linewidth=0.5)

fig.legend(title='Tipo de frontera')

ax[3].plot(model.vol, linewidth=0.5)

ax[3].set_xlabel('Paso de tiempo')
ax[3].set_ylabel('Volumen [m³]')

plt.show()
