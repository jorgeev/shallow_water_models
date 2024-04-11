# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:53:23 2024

@author: jevza
"""

from myfirstshallowwatermodel.model1d_numba import *

model_data = init()
for ii in range(model_data['nt']):
    if ii < model_data['period']:
        model_data['h1'] = perturbate(model_data['h1'], ii, model_data['DT'], model_data['period'])
    else:
        model_data['h1'], model_data['u1'] = periodic_boundary_conditions(model_data['h1'], model_data['u1'])
        
    if tt == 0:
        model_data['u1'] = u_forward(model_data['u1'], model_data['u1'], model_data['h1'], model_data['mu0u'], model_data['X'])
        model_data['h1'] = h_forward(model_data['h0'], model_data['h1'], model_data['u1'], model_data['mu0h'], model_data['X'])
    else:
        model_data['u2'] = u_centerd(model_data['u2'], model_data['u0'], model_data['h1'], model_data['muxu'], model_data['X'])
        model_data['h2'] = h_centerd(model_data['h2'], model_data['h0'], model_data['u1'], model_data['muxh'], model_data['X'])   
    
    model_data['u0'] = model_data['u1'].copy()
    model_data['u1'] = model_data['u2'].copy()
    model_data['h0'] = model_data['h1'].copy()
    model_data['h1'] = model_data['h2'].copy()
    
