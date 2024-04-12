#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:48:56 2024

@author: jorgeev@github.com
Example for the model configuration
"""

from myfirstshallowwatermodel.model1d import simple1D as sw1d
import numpy as np
import matplotlib.pyplot as plt

model = sw1d(X = 150, boundary="p", nt=400, 
             plotting=False, plot_interval=10, use_asselin=False,
             asselin_step=1, asselin_value=0.01,
             save_outputs=True, save_interval=100
             )

model.run()

fig, ax = plt.subplots(1,1,figsize=(6,8),sharex=True, dpi=300)
ax.plot(model.h_output.T)
plt.show()

H_outputs = np.flipud(model.h_output[1:].T)
H_mean = np.mean(H_outputs, axis=0)
H_std = np.std(H_outputs, axis=0)
fig, ax = plt.subplots(1,1,figsize=(6,8),sharex=True, dpi=300)
ax.pcolormesh(H_outputs)
plt.show()

H_norm = (H_outputs - H_mean)/H_std
C = 1/4 * np.matmul(H_norm[:,1:], H_norm[:,1:].T)
H_norm[:,0] = 0
fig, ax = plt.subplots(1,1,figsize=(6,8),sharex=True, dpi=300)
ax.pcolormesh(H_norm)
plt.show()


fig, ax = plt.subplots(1,1,figsize=(6,8),sharex=True, dpi=300)
ax.pcolormesh(C)
plt.show()

eigenvalues, eigenvectors = np.linalg.eig(C)

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