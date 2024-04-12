# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:49:24 2024

@author: je.velasco
"""

import numpy as np
import numba as nb
from numba.core import types as nbtypes

def init(
        X:int=100, 
        DX:float=1000., 
        H_0:float=1500, 
        DT:float=1, 
        nt:int=1000,
        g:float=9.81,
        period:int=450
        ):
    H = np.ones(X) * H_0
    mu0u = g * DT / DX
    mu0h = H * DT / DX
    return dict(
        X=X, 
        DX=DX, 
        H_0=H_0, 
        DT=DT,
        g=g, 
        period=period,
        nt=nt,
        H = H,
        h0 = np.zeros(X),
        h1 = np.zeros(X),
        h2 = np.zeros(X),
        u0 = np.zeros(X+1),
        u1 = np.zeros(X+1),
        u2 = np.zeros(X+1),
        E_k = np.zeros(nt),
        E_p = np.zeros(nt),
        vol = np.zeros(nt),
        mu0u = mu0u,
        mu0h = mu0h,
        muxu = 2 * mu0u,
        muxh = 2 * mu0h,
        )
@nb.njit(nb.float64[:](nb.float64[:], nb.int8, nb.int8, nb.int8))
def perturbate(h1, ii, DT, period):
        h1[0] = np.sin((ii)*2*np.pi*DT/period)
        h1[1] = np.sin((ii-1)*2*np.pi*DT/period)
        return h1

@nb.njit(nbtypes.Tuple([nb.float64[:], nb.float64[:]])(nb.float64[:],nb.float64[:]))
def periodic_boundary_conditions(h1, u1):
    h1[0] = h1[-2]
    h1[-1] = h1[1]
    u1[0] = u1[-2]
    u1[-1] = u1[1]
    return h1, u1

@nb.njit(nb.float64[:](nb.float64[:]))
def closed__boundary_conditions(u1):
    u1[-2] = 0
    return u1

@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.int8), parallel=True)
def u_forward(u0, u1, h1, mu0u, X):
    for ii in nb.prange(1,X):
        u1[ii] = u0[ii] - mu0u * (h1[ii] - h1[ii-1])
    return u1

@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.int8), parallel=True)
def h_forward(h0, h1, u1, mu0h, X):
    for ii in nb.prange(1, X):
        h1[ii] = h0[ii] - mu0h[ii] *(u1[ii+1] - u1[ii])
    return h1

@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.int8), parallel=True)
def u_centered(u2, u0, h1, muxu, X):
    for ii in nb.prange(1, X):
        u2[ii] = u0[ii] - muxu * (h1[ii] - h1[ii-1])
    return u2

@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.int8), parallel=True)
def h_centered(h2, h0, u1, muxh, X):
    for ii in nb.prange(1, X):
        h2[ii] = h0[ii] - muxh[ii] * (u1[ii+1] - u1[ii])
    return h2


    