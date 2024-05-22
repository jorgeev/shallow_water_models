# -*- coding: utf-8 -*-
"""
Created on Wed May 22 08:29:32 2024
Simulation created for the Atmospheric Dynamics class
@author: je.velasco@atmosfera.unam.mx
"""

from myfirstshallowwatermodel.model2d import simple2dmodel as sw2d
from netCDF4 import Dataset
import matplotlib.pyplot as plt

model2 = sw2d(X=500, Y=500, H_0=10000, nt=2592000, nesting=False, DT=1,
              omega=7.29E-5, origin=(250,250), initialc='e', use_asselin=True,
              asselin_value=0.1, asselin_step=1, calculate_metrics=True,
              DX=1000, DY=1000, plotting=False, plot_path='Quintanar',
              save_interval=1800, size=(6,6), maxh0=1., lat=45., 
              save_outputs=True)

model2.run(cmap='twilight_shifted')

vault = Dataset('Quintanar.nc', 'w', format='NETCDF4')
vault.createDimension("x", 501)
vault.createDimension("y", 501)
vault.createDimension("xc", 500)
vault.createDimension("yc", 500)
vault.createDimension("time", int(1+500/100))
#time = vault.createVariable("time", "u8", ("time"))
hh = vault.createVariable("h", "f8", ("time", 'yc', 'xc'))
uu = vault.createVariable("u", "f8", ("time", 'yc', 'x'))
vv = vault.createVariable("v", "f8", ("time", 'y', 'xc'))
hh[:] = model2.h_output
uu[:] = model2.u_output
vv[:] = model2.v_output
vault.close()
