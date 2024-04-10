#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:48:56 2024

@author: jorgeev@github.com
Example for the model configuration
"""

from myfirstshallowwatermodel.model1d import simple1D as sw1d
import matplotlib.pyplot as plt

model = sw1d(X = 150, boundary="p", nt=3000, 
             plotting=True, plot_interval=10, use_asselin=True,
             asselin_step=100, asselin_value=0.01,
             )

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