#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:34:10 2024

@author: jevz
"""
from os import mkdir
from os.path import isdir
import numpy as np
import matplotlib.pyplot as plt

class simple1D:

    def __init__(self, X:int=150, DX:float=1000, DT:float=3, nt:int=3000, H_0:float=1500, 
                 period:float=450, gravity:float=9.81, boundary:str="p", 
                 use_asselin:bool=False, asselin_value:float=0.1, asselin_step:int=1,
                 plotting:bool=False, plot_path:str='1D_output', plot_interval:int=100,
                 save_outputs:bool=False, save_interval:int=100,
                 ):
        """
        X       : mesh size
        DX      : mesh spacing [m]
        DT      : time step [s]
        H_0     : Average depth 
        nt      : number of time steps
        period  : initial perturbation period (s)
        g       : gravity m/s^2
        boundary: "p" = periodic, "c" = closed
        """
        self.X = X
        self.DX = DX
        self.DT = DT
        self.nt = nt+1
        self.period = period
        self.gravity = gravity
        self.v1 = gravity * DT / DX
        self.v1_ = self.v1 * 2
        self.H = np.ones(X) * H_0
        self.v2 = self.H * DT / DX
        self.v2_ = self.v2 * 2
        self.h0 = np.zeros(X) # Auxiliar vectors
        self.h1 = np.zeros(X)
        self.h2 = np.zeros(X)
        self.u0 = np.zeros(X+1)
        self.u1 = np.zeros(X+1)
        self.u2 = np.zeros(X+1)
        self.E_k = np.zeros(self.nt)
        self.E_p = np.zeros(self.nt)
        self.vol = np.zeros(self.nt)
        self.boundary = boundary
        
        # Asselin parameters
        self.asselin = use_asselin
        self.asselin_coef = asselin_value
        self.asselin_step = asselin_step
        # Plotting parameters
        self.plotting = plotting
        self.path = plot_path
        self.plot_interval = plot_interval
        # Save outputs
        self.save_outputs = save_outputs
        self.save_interval = save_interval
        self.u_output = np.zeros([int(self.nt/self.save_interval)+1, self.X+1])
        self.h_output = np.zeros([int(self.nt/self.save_interval)+1, self.X])
        
    def check_dir(self):
        if isdir(self.path) == False:
            mkdir(self.path)

    def apply_boundary_conditions(self, ii):
        if ii + 1 <= (self.period/self.DT + 1):
            self.h1[0] = np.sin((ii)*2*np.pi*self.DT/self.period)
            self.h1[1] = np.sin((ii-1)*2*np.pi*self.DT/self.period)
        else:
            if self.boundary == "p":
                self.h1[0] = self.h1[-2]
                self.h1[-1] = self.h1[1]
                self.u1[0] = self.u1[-2]
                self.u1[-1] = self.u1[1]

            if self.boundary == "c":
                self.u1[-2] = 0
            
    
    def forward_differece(self):
        self.u1[1:-1] = self.u0[1:-1] - self.v1 * (self.h1[1:] - self.h1[:-1])
        self.h1[1:] = self.h0[1:] - self.v2[1:] * (self.u1[2:] - self.u1[1:-1])
        
    
    def centered_difference(self):
        self.u2[1:-1] = self.u0[1:-1] - self.v1_ * (self.h1[1:] - self.h1[:-1])
        self.h2[1:] = self.h0[1:] - self.v2_[1:] * (self.u1[2:] - self.u1[1:-1])

    
    def get_kinetic_energy(self,ii):
        """
        E_k = H/2 * SUM(u2^2 * dx)
        """
        self.E_k[ii] = 0.5 * np.sum(self.H * self.u2[:-1] ** 2) #* self.DX
    
    def get_potential_energy(self,ii):
        """
        E_p = g/2 * SUM(h2^2 * dx)
        """
        self.E_p[ii] = 0.5 * self.gravity * np.sum(self.h2 ** 2)  #* self.DX
        
    
    def calc_volume(self, ii):
        self.vol[ii] = np.sum(np.abs(self.h2)) * self.DX
    
    def apply_asselin(self):
        self.u1 += self.asselin_coef * (self.u0 - 2*self.u1 + self.u2)
        self.h1 += self.asselin_coef * (self.h0 - 2*self.h1 + self.h2)
        
    def save_figures(self, tt):
        fig, (ax1,ax2)= plt.subplots(2,1,figsize=(6,4),sharex=True)
        ax1.plot(self.h2)
        ax2.plot(self.u2,'r')
        ax1.set_title('h al paso de tiempo %d' %tt)
        ax2.set_title('u al paso de tiempo %d' %tt)
        ax2.set_xlabel('índice x')
        ax1.set_ylabel('h')
        ax2.set_ylabel('u')
        ax1.set_xlim(0,self.X)
        ax1.set_ylim(-1.5,1.5)
        ax2.set_ylim(-0.20,0.20)
        ax1.grid()
        ax2.grid()
        plt.savefig(F'{self.path}/{tt:06d}.jpg', bbox_inches='tight', pad_inches=0.1)
        plt.cla()
        plt.clf()
        plt.close(fig)

        
    def run(self):
        self.check_dir()
        self.outputs=0
        for tt in range(self.nt):
            self.apply_boundary_conditions(tt)
            if tt == 0:
                self.forward_differece()
            else:
                self.centered_difference()
            
            if self.asselin and tt!=0:
                self.apply_asselin()
            
            self.u0 = self.u1.copy()
            self.u1 = self.u2.copy()
            self.h0 = self.h1.copy()
            self.h1 = self.h2.copy()
            
            self.get_kinetic_energy(tt)
            self.get_potential_energy(tt)
            self.calc_volume(tt)
            
            if tt!=0 and self.plotting and tt%self.plot_interval==0:
                self.save_figures(tt)
                
            if self.save_outputs and tt%self.save_interval==0:
                self.u_output[self.outputs, :] = self.u2.copy()
                self.h_output[self.outputs, :] = self.h2.copy()
                self.outputs+=1
                
class windstress_1D:

    def __init__(self, X:int=150, DX:float=1000, DT:float=3, nt:int=3000, H_0:float=1500, 
                 wind_duration:float=450, wind_speed=-0.01, gravity:float=9.81, boundary:str="p", 
                 use_asselin:bool=False, asselin_value:float=0.1, asselin_step:int=1,
                 plotting:bool=False, plot_path:str='1D_output', plot_interval:int=100,
                 save_outputs:bool=False, save_interval:int=100, rho_h2o:float=1025.
                 ):
        """
        X       : mesh size
        DX      : mesh spacing [m]
        DT      : time step [s]
        H_0     : Average depth 
        nt      : number of time steps
        period  : initial perturbation period (s)
        g       : gravity m/s^2
        boundary: "p" = periodic, "c" = closed
        """
        self.X = X
        self.DX = DX
        self.DT = DT
        self.nt = nt+1
        self.gravity = gravity
        self.v1 = gravity * DT / DX
        self.v1_ = self.v1 * 2
        self.H = np.ones(X) * H_0
        self.v2 = self.H * DT / DX
        self.v2_ = self.v2 * 2
        self.h0 = np.zeros(X) # Auxiliar vectors
        self.h1 = np.zeros(X)
        self.h2 = np.zeros(X)
        self.u0 = np.zeros(X+1)
        self.u1 = np.zeros(X+1)
        self.u2 = np.zeros(X+1)
        self.E_k = np.zeros(self.nt)
        self.E_p = np.zeros(self.nt)
        self.vol = np.zeros(self.nt)
        self.boundary = boundary
        self.windwind_duration = wind_duration
        self.wind_speed = wind_speed
        self.rho_h2o = rho_h2o
        self.m1 = self.DT / self.rho_h2o
        
        # Asselin parameters
        self.asselin = use_asselin
        self.asselin_coef = asselin_value
        self.asselin_step = asselin_step
        # Plotting parameters
        self.plotting = plotting
        self.path = plot_path
        self.plot_interval = plot_interval
        # Save outputs
        self.save_outputs = save_outputs
        self.save_interval = save_interval
        self.u_output = np.zeros([int(self.nt/self.save_interval)+1, self.X+1])
        self.h_output = np.zeros([int(self.nt/self.save_interval)+1, self.X])
        
    def check_dir(self):
        if isdir(self.path) == False:
            mkdir(self.path)

    def apply_boundary_conditions(self):
            if self.boundary == "p":
                self.h1[0] = self.h1[-2]
                self.h1[-1] = self.h1[1]
                self.u1[0] = self.u1[-2]
                self.u1[-1] = self.u1[1]

            if self.boundary == "c":
                self.u1[-1] = 0
                self.u1[0] = 0
                self.u1[1] = 0
                self.u1[-2]*=0.1
                self.u1[-3]*=0.2
                self.u1[-4]*=0.3
                self.u1[-5]*=0.4
                self.u1[-6]*=0.5
                self.u1[-7]*=0.6
                self.u1[-8]*=0.7
                self.u1[-9]*=0.8
                self.u1[-10]*=0.9
                
                
    
    def forward_differece(self):
        self.u1[1:-1] = self.u0[1:-1] - self.v1 * (self.h1[1:] - self.h1[:-1]) + self.wind_speed * self.m1
        self.h1[1:] = self.h0[1:] - self.v2[1:] * (self.u1[2:] - self.u1[1:-1])
        
    
    def centered_difference(self):
        self.u2[1:-1] = self.u0[1:-1] - self.v1_ * (self.h1[1:] - self.h1[:-1]) + self.wind_speed * self.m1
        self.h2[1:] = self.h0[1:] - self.v2_[1:] * (self.u1[2:] - self.u1[1:-1])

    
    def get_kinetic_energy(self,ii):
        """
        E_k = H/2 * SUM(u2^2 * dx)
        """
        self.E_k[ii] = 0.5 * np.sum(self.H * self.u2[:-1] ** 2) #* self.DX
    
    def get_potential_energy(self,ii):
        """
        E_p = g/2 * SUM(h2^2 * dx)
        """
        self.E_p[ii] = 0.5 * self.gravity * np.sum(self.h2 ** 2)  #* self.DX
        
    
    def calc_volume(self, ii):
        self.vol[ii] = np.sum(np.abs(self.h2)) * self.DX
    
    def apply_asselin(self):
        self.u1 += self.asselin_coef * (self.u0 - 2*self.u1 + self.u2)
        self.h1 += self.asselin_coef * (self.h0 - 2*self.h1 + self.h2)
        
    def save_figures(self, tt):
        fig, (ax1,ax2)= plt.subplots(2,1,figsize=(6,4),sharex=True)
        ax1.plot(self.h2)
        ax2.plot(self.u2,'r')
        ax1.set_title('h al paso de tiempo %d' %tt)
        ax2.set_title('u al paso de tiempo %d' %tt)
        ax2.set_xlabel('índice x')
        ax1.set_ylabel('h')
        ax2.set_ylabel('u')
        ax1.set_xlim(0,self.X)
        ax1.set_ylim(-1.5,1.5)
        ax2.set_ylim(-0.20,0.20)
        ax1.grid()
        ax2.grid()
        plt.savefig(F'{self.path}/{tt:06d}.jpg', bbox_inches='tight', pad_inches=0.1)
        plt.cla()
        plt.clf()
        plt.close(fig)

        
    def run(self):
        self.check_dir()
        self.outputs=0
        for tt in range(self.nt):
            self.apply_boundary_conditions()
            if tt == self.windwind_duration:
                self.m1 = 0.
                print(tt, 'wind_off')
            if tt == 0:
                self.forward_differece()
            else:
                self.centered_difference()
            
            if self.asselin and tt!=0:
                self.apply_asselin()
            
            self.u0 = self.u1.copy()
            self.u1 = self.u2.copy()
            self.h0 = self.h1.copy()
            self.h1 = self.h2.copy()
            
            self.get_kinetic_energy(tt)
            self.get_potential_energy(tt)
            self.calc_volume(tt)
            
            if tt!=0 and self.plotting and tt%self.plot_interval==0:
                self.save_figures(tt)
                
            if self.save_outputs and tt%self.save_interval==0:
                self.u_output[self.outputs, :] = self.u2.copy()
                self.h_output[self.outputs, :] = self.h2.copy()
                self.outputs+=1