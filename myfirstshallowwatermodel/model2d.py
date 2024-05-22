# -*- coding: utf-8 -*-
"""
Swallow water model in 2D with support to host a nesting inside the domain
version 2024.04.24T09
author jorgeev@github.com
"""

import numpy as np
import scipy as cp
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits import mplot3d
from os import mkdir
from os.path import isdir
from netCDF4 import Dataset

class simple2dmodel:
    
    def __init__(self, X:int=200, DX:float=1000, 
                 Y:int=200, DY:float=1000, 
                 DT:float=3., nt:int=3000, 
                 H_0:float=1500., gravity:float=9.81, 
                 initialc:str="c", omega:float=7.29E-5, lat:float=30, 
                 period:float = 5000, 
                 nesting:bool=False, nest_ratio:float=3, 
                 use_asselin:bool=False, asselin_value:float=0.1, asselin_step:int=1, 
                 calculate_metrics:bool=False,
                 nestpos:tuple=(150,150,30,30), # tuple(x_origin, y_origin, width, height)
                 dampening:int=10, plotting:bool=False, plot_path='sim_output', plot_interval:int=100,
                 origin:tuple=(100,100), size:tuple=(2,2), maxh0:float=1.,
                 save_outputs:bool=False, save_interval:int = 500):
        """
        X, Y    : mesh size
        DX, DY  : mesh spacing [m]
        DT      : time step [s]
        H_0     : Average depth 
        nt      : number of time steps
        period  : initial perturbation period (s)
        g       : gravity m/s^2
        condition: "c" = cosine_condition, "e" = exponential_conditon, 'D' = centered initial condition as in original code
        origin: centering for the initial condition only used if condition is exponential and cos
        size: relative size for the initial condition exponential only
        """
        self.use_nest = nesting
        self.X = X
        self.DX = DX
        self.Y = Y
        self.DY = DY
        self.DT = DT
        self.nt = nt
        self.gravity = gravity
        self.H = np.ones((Y, X)) * H_0
        self.condition = initialc
        self.h0 = np.zeros((Y, X))
        self.h1 = np.zeros((Y, X))
        self.h2 = np.zeros((Y, X))
        self.u0 = np.zeros((Y, X+1))
        self.u1 = np.zeros((Y, X+1))
        self.u2 = np.zeros((Y, X+1))
        self.v0 = np.zeros((Y+1, X))
        self.v1 = np.zeros((Y+1, X))
        self.v2 = np.zeros((Y+1, X))
        self.CFL = np.sqrt(self.gravity*np.nanmax(self.H)) * self.DT/self.DX
        self.f = 2 * omega * np.sin(np.deg2rad(lat))
        
        self.Xv = np.arange(self.X)
        self.Yv = np.arange(self.Y)
        self.x_p, self.y_p = np.meshgrid(self.Xv, self.Yv)
        
        self.originX = origin[0]
        self.originY = origin[1]
        self.size = size
        self.B = maxh0
        
        self.plotting = plotting
        self.path = plot_path
        self.plot_interval = plot_interval
        
        # Asselin parameters
        self.asselin = use_asselin
        self.asselin_coef = asselin_value
        self.asselin_step = asselin_step
        
        # for metrics calculation
        self.metrics = calculate_metrics
        if self.metrics:
            self.E_k = np.zeros(nt)
            self.E_p = np.zeros(nt)
            self.vol = np.zeros(nt)
            
        # to save outputs
        self.save_outputs = save_outputs
        if self.save_outputs:
            self.save_interval = save_interval
            self.u_output = np.zeros([int(self.nt/self.save_interval)+1, self.Y, self.X+1])
            self.v_output = np.zeros([int(self.nt/self.save_interval)+1, self.Y+1, self.X])
            self.h_output = np.zeros([int(self.nt/self.save_interval)+1, self.Y, self.X])
        
        # for nested data
        if self.use_nest:
            self.nest_ratio = nest_ratio
            self.roi = nestpos
            self.dampening = dampening
            self.Xn = nestpos[2] * nest_ratio + 1 
            self.Yn = nestpos[3] * nest_ratio + 1 
            self.DTn = self.DT / nest_ratio
            self.DXn = self.DX / nest_ratio
            self.DYn = self.DY / nest_ratio
            self.x0nest = nestpos[0]
            self.x1nest = nestpos[0] + nestpos[2] + 1
            self.y0nest = nestpos[1]
            self.y1nest = nestpos[1] + nestpos[3] + 1
            
            self.u_1 = np.zeros([self.Yn, self.Xn + 1])
            self.u_2 = np.zeros([self.Yn, self.Xn + 1])
            self.v_1 = np.zeros([self.Yn + 1, self.Xn])
            self.v_2 = np.zeros([self.Yn + 1, self.Xn])
            self.h_1 = np.zeros([self.Yn, self.Xn])
            self.h_2 = np.zeros([self.Yn, self.Xn])
            # Parametros para la interpolacion de dominios
            self.xh_xa = np.arange(self.X)
            self.yh_xa = np.arange(self.Y)
            self.xu_xa = np.arange(-0.5,self.X+0.5)*self.DX
            self.yv_xa = np.arange(-0.5,self.Y+0.5)*self.DY
            self.t0 = np.arange(2)        
            self.xh_tg = np.linspace(self.xh_xa[self.x0nest], self.xh_xa[self.x1nest], self.Xn)
            self.yh_tg = np.linspace(self.yh_xa[self.y0nest], self.yh_xa[self.y1nest], self.Yn)
            self.xu_tg = np.linspace(self.xu_xa[self.x0nest], self.xu_xa[self.x1nest], self.Xn+1)
            self.yv_tg = np.linspace(self.yv_xa[self.y0nest], self.yv_xa[self.y1nest], self.Yn+1)
            self.tt_tg = np.linspace(0, 1, self.nest_ratio)
            # need to add seving outpust in nesting
            
    def check_dir(self):
        if isdir(self.path) == False:
            mkdir(self.path)
            
    def create_perturbation0(self):
        z = self.h0.copy()
        for jj in range(int(self.Y/2-0.2*(self.Y/2)), int(self.Y/2 + 0.2*(self.Y/2)+1)):
            for ii in range(int(self.X/2-0.2*(self.X/2)), int(self.X/2 + 0.2*(self.X/2)+1)):
                r = np.sqrt((ii-np.round(self.X/2))**2 + (jj-np.round(self.Y/2))**2)
                if r <=0.2*(self.X/2):
                    z[jj,ii] = 0.5*(1+np.cos(np.pi*r/(0.2*(self.X/2))))
        self.h0 = z.copy()
     
    
    def create_perturbation1(self):
        z = self.h0.copy()
        for jj in range(int(self.originY-0.2*(self.originY)), int(self.originY + 0.2*(self.originY)+1)):
            for ii in range(int(self.originX-0.2*(self.originX)), int(self.originX + 0.2*(self.originX)+1)):
                r = np.sqrt((ii-np.round(self.originX))**2 + (jj-np.round(self.originY))**2)
                if r <=0.2*self.originX and r <=0.2*self.originY:
                    z[jj,ii] = 0.5*(1+np.cos(np.pi*r/(0.2*(self.X/2))))
        self.h0 = z.copy()
    
    def create_perturbation2(self):
        sigma_x = 0.2*self.X/self.size[0]
        sigma_y = 0.2*self.X/self.size[1] 
        z = self.B * np.exp(-((self.x_p-self.originX)**2/(2*sigma_x**2) + (self.y_p-self.originY)**2/(2*sigma_y**2)))
        z = np.nan_to_num(z)
        self.h0 = z.copy()
    
    def forward_difference(self):
        self.u1[:,1:-1] = self.u0[:,1:-1] -  self.gravity * self.DT/self.DX * (self.h0[:,1:] - self.h0[:,:-1]) + self.f * self.DT * (self.v1[1:,1:] + self.v1[1:,:-1] + self.v1[:-1,1:] + self.v1[:-1,:-1])/4
        self.v1[1:-1,:] = self.v0[1:-1,:] -  self.gravity * self.DT/self.DY * (self.h0[1:,:] - self.h0[:-1,:]) - self.f * self.DT * (self.u1[1:,1:] + self.u1[:-1,1:] + self.u1[1:,:-1] + self.u1[:-1,:-1])/4
        self.h1[1:-1, 1:-1] = self.h0[1:-1,1:-1] - self.H[1:-1,1:-1] * (self.DT/self.DX * (self.u0[1:-1,2:-1] - self.u0[1:-1,1:-2]) + self.DT/self.DY * (self.v0[2:-1,1:-1] - self.v0[1:-2,1:-1]))   
        
    def centered_differences(self):
        self.u2[:,1:-1] = self.u0[:,1:-1] - 2 * self.gravity * self.DT/self.DX * (self.h1[:,1:] - self.h1[:,:-1]) + self.f * self.DT * (self.v1[1:,1:] + self.v1[1:,:-1] + self.v1[:-1,1:] + self.v1[:-1,:-1])/2
        self.v2[1:-1,:] = self.v0[1:-1,:] - 2 * self.gravity * self.DT/self.DY * (self.h1[1:,:] - self.h1[:-1,:]) - self.f * self.DT * (self.u1[1:,1:] + self.u1[:-1,1:] + self.u1[1:,:-1] + self.u1[:-1,:-1])/2
        self.h2[1:-1, 1:-1] = self.h0[1:-1,1:-1] - 2 * self.H[1:-1,1:-1] * (self.DT/self.DX * (self.u1[1:-1,2:-1] - self.u1[1:-1,1:-2]) + self.DT/self.DY * (self.v1[2:-1,1:-1] - self.v1[1:-2,1:-1]))
        
    def centered_differences_nest(self):
        self.u_2[:,1:-1] = self.u_0[:,1:-1] - 2 * self.gravity * self.DTn/self.DXn * (self.h_1[:,1:] - self.h_1[:,:-1]) + self.f * self.DTn * (self.v_1[1:,1:] + self.v_1[1:,:-1] + self.v_1[:-1,1:] + self.v_1[:-1,:-1])/4
        self.v_2[1:-1,:] = self.v_0[1:-1,:] - 2 * self.gravity * self.DTn/self.DYn * (self.h_1[1:,:] - self.h_1[:-1,:]) - self.f * self.DTn * (self.u_1[1:,1:] + self.u_1[:-1,1:] + self.u_1[1:,:-1] + self.u_1[:-1,:-1])/4
        self.h_2[1:-1, 1:-1] = self.h_0[1:-1,1:-1] - 2 * self.H2[1:-1,1:-1] * self.DTn/self.DYn * (self.u_1[1:-1,2:-1] - self.u_1[1:-1,1:-2] + self.v_1[2:-1,1:-1] - self.v_1[1:-2,1:-1])
    
    def create_masks(self):
        Is = self.dampening
        mu = 0.5 * (1+np.cos(np.pi*np.arange(0,Is + 1)/Is ))
        self.hmask = np.zeros((self.Yn, self.Xn))
        self.vmask = np.zeros((self.Yn+1, self.Xn))
        self.umask = np.zeros((self.Yn, self.Xn+1))
        
        self.hmask[:Is+1, :] = mu.reshape(Is+1,1)
        self.hmask[-Is-1:, :] = np.flipud(mu.reshape(Is+1,1))
        self.umask[:Is+1, :] = mu.reshape(Is+1,1)
        self.umask[-Is-1:, :] = np.flipud(mu.reshape(Is+1,1))
        self.vmask[:Is+1, :] = mu.reshape(Is+1,1)
        self.vmask[-Is-1:, :] = np.flipud(mu.reshape(Is+1,1))
        
        for ii in range(Is):
            for jj in range(ii+1, self.Yn-ii):
                self.hmask[jj, ii] = 0.5 * (1+np.cos(np.pi*ii/Is ))
                self.hmask[jj,-ii] = 0.5 * (1+np.cos(np.pi*ii/Is ))
                self.umask[jj, ii] = 0.5 * (1+np.cos(np.pi*ii/Is ))
                self.umask[jj,-ii] = 0.5 * (1+np.cos(np.pi*ii/Is ))
                self.vmask[jj, ii] = 0.5 * (1+np.cos(np.pi*ii/Is ))
                self.vmask[jj,-ii] = 0.5 * (1+np.cos(np.pi*ii/Is ))
    
    def interp_topo(self):
        self.H2 = xr.DataArray(self.H, dims=['y', 'x'], coords={'x':(('x'), np.arange(self.X) * self.DX),
                                                                'y':(('y'), np.arange(self.Y) * self.DY)})
        self.xn = np.linspace(self.H2.x[self.x0nest], self.H2.x[self.x1nest], self.Xn)
        self.yn = np.linspace(self.H2.y[self.y0nest], self.H2.y[self.y1nest], self.Yn)

        self.H2 = self.H2.interp(x=self.xn, y=self.yn)
    
    def interp_ic(self):
        h = xr.DataArray(self.h1, dims=['y','x'], coords={'x':(('x'), self.xh_xa),
                                                          'y':(('y'), self.yh_xa)})
        u = xr.DataArray(self.u1, dims=['y','x'], coords={'x':(('x'), self.xu_xa),
                                                          'y':(('y'), self.yh_xa)})
        v = xr.DataArray(self.v1, dims=['y','x'], coords={'x':(('x'), self.xh_xa),
                                                          'y':(('y'), self.yv_xa)})
        self.u_0 = u.interp(x=self.xu_tg, y=self.yh_tg).data.copy()
        self.v_0 = v.interp(x=self.xh_tg, y=self.yv_tg).data.copy()
        self.h_0 = h.interp(x=self.xh_tg, y=self.yh_tg).data.copy()
    
    def interpn_nesting(self):
        h1 = np.stack([self.h0, self.h1], axis=-1)
        h2 = np.stack([self.h1, self.h2], axis=-1)
        u1 = np.stack([self.u0, self.u1], axis=-1)
        u2 = np.stack([self.u1, self.u2], axis=-1)
        v1 = np.stack([self.v0, self.v1], axis=-1)
        v2 = np.stack([self.v1, self.v2], axis=-1)
        
        self.hxr = xr.Dataset(
            data_vars=dict(h1=(('y','x','t'), h1), h2=(('y','x','t'), h2)),
            coords={'x':(('x'), self.xh_xa), 
                    'y':(('y'), self.yh_xa),
                    't':(('t'), self.t0)})
        self.uxr = xr.Dataset(
            data_vars=dict(u1=(('y','x','t'), u1), u2=(('y','x','t'), u2)),
            coords={'x':(('x'), self.xu_xa), 
                    'y':(('y'), self.yh_xa),
                    't':(('t'), self.t0)})
        self.vxr = xr.Dataset(
            data_vars=dict(v1=(('y','x','t'), v1), v2=(('y','x','t'), v2)),
            coords={'x':(('x'), self.xh_xa), 
                    'y':(('y'), self.yv_xa),
                    't':(('t'), self.t0)})
               
        self.hxr = self.hxr.interp(x=self.xh_tg, y=self.yh_tg, t=self.tt_tg)
        self.uxr = self.uxr.interp(x=self.xu_tg, y=self.yh_tg, t=self.tt_tg)
        self.vxr = self.vxr.interp(x=self.xh_tg, y=self.yv_tg, t=self.tt_tg)

    def save_figures(self, tt, cmap='viridis'): 
        fig = plt.figure(figsize=(13,6), dpi=300)
        fig.suptitle(F'CFL:{self.CFL:06f}, timestep:{tt}', fontsize=16)
        if self.use_nest:
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3, projection='3d')
            ax4 = fig.add_subplot(2, 2, 4)
            roi = patches.Rectangle((self.roi[0], self.roi[1]), self.roi[2], self.roi[3], 
                                    linewidth=1, edgecolor='r', facecolor='none', zorder=100)
            ax2.add_patch(roi)
            
            ax3.set_xlabel('y (km)')
            ax3.set_ylabel('x (km)')
            ax3.set_zlabel('z (m)')
            ax4.set_xlabel('y (km)')
            ax4.set_ylabel('x (km)')
        else:
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2)
        
        if tt == 0:
            ax1.plot_surface(self.x_p[1:-1,1:-1], self.y_p[1:-1,1:-1], self.h1[1:-1,1:-1], 
                             cmap=cmap, edgecolor='none', antialiased=True, vmin=-0.5, vmax=0.5)
            img = ax2.pcolormesh(self.x_p, self.y_p, self.h1, 
                                 vmin=-0.5, vmax=0.5, cmap=cmap)
            if self.use_nest:
                ax4.pcolormesh(self.xnm/1000, self.ynm/1000, self.h_1, 
                               vmin=-0.5, vmax=0.5, cmap=cmap)
                ax3.plot_surface(self.xnm/1000, self.ynm/1000, self.h_1, 
                                 cmap=cmap, edgecolor='none', antialiased=True, vmin=-0.5, vmax=0.5)
                ax4.set_aspect(1)
        else:
            ax1.plot_surface(self.x_p[1:-1,1:-1], self.y_p[1:-1,1:-1], self.h2[1:-1,1:-1], 
                             cmap=cmap, edgecolor='none', antialiased=True, vmin=-0.5, vmax=0.5)
            img = ax2.pcolormesh(self.x_p, self.y_p, self.h2, 
                                 vmin=-0.5, vmax=0.5, cmap=cmap)
            if self.use_nest:
                ax4.pcolormesh(self.xnm/1000, self.ynm/1000, self.h_2, 
                               vmin=-0.5, vmax=0.5, cmap=cmap)
                ax3.plot_surface(self.xnm/1000, self.ynm/1000, self.h_2, 
                                 cmap=cmap, edgecolor='none', antialiased=True, vmin=-0.5, vmax=0.5)
                ax4.set_aspect(1)
            
            
        ax2.set_aspect(1)        
        ax1.set_xlabel('y (km)')
        ax1.set_ylabel('x (km)')
        ax1.set_zlabel('z (m)')
        ax2.set_xlabel('y (km)')
        ax2.set_ylabel('x (km)')
          
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
        fig.colorbar(img, cax=cbar_ax, label='$h$ (m)')
        plt.savefig(F'{self.path}/{tt:06d}.jpg', bbox_inches='tight', pad_inches=0.1)
        plt.cla()
        plt.clf()
        plt.close(fig)
    
    def apply_asselin(self):
        self.u1 += self.asselin_coef * (self.u0 - 2*self.u1 + self.u2)
        self.v1 += self.asselin_coef * (self.v0 - 2*self.v1 + self.v2)
        self.h1 += self.asselin_coef * (self.h0 - 2*self.h1 + self.h2)
    
    def update_uvh(self, tt):
        if tt!=0:
            self.u0 = self.u1.copy()
            self.v0 = self.v1.copy()
            self.h0 = self.h1.copy()
            self.u1 = self.u2.copy()
            self.v1 = self.v2.copy()
            self.h1 = self.h2.copy()
        else:
            self.u0 = self.u1.copy()
            self.v0 = self.v1.copy()
            self.h0 = self.h1.copy()
       
    def boundary_conditions(self):
       self.u1[:,1]  = 0
       self.u1[:,-2] = 0
       self.v1[1,:]  = 0
       self.v1[-2,:] = 0
       
    def plot_speed(self,tt):
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,6), dpi=300)
        pc1 = ax1.pcolormesh(self.h2, vmin=-0.5, vmax=0.5, cmap='PRGn')
        plt.colorbar(pc1, ax=ax1,label='h (m)')
        ax1.set_ylabel('y')
        ax1.axhline(100,linestyle='--', color='0.5')
        ax1.set_aspect(1)

        ax2.plot(self.h2[150,:], 'g-')
        ax2.plot(self.u2[150,:], 'r-')
        ax2.plot(self.v2[150,:], 'b-')
        ax2.set_ylabel('h (m)')
        #ax2.set_ylim(-0.5,0.5)

        pc3 = ax3.pcolormesh(self.u2, cmap='RdBu')#, vmin=-1E-3, vmax=1E-3,)
        plt.colorbar(pc3, ax=ax3,label='u (m/s)')
        ax3.set_xlabel('x')
        ax3.axhline(100,linestyle='--', color='0.5')
        ax3.set_aspect(1)

        pc4 = ax4.pcolormesh(self.v2, cmap='RdBu')#, vmin=-1E-3, vmax=1E-3,)
        plt.colorbar(pc4, ax=ax4,label='v (m/s)')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.axhline(100,linestyle='--', color='0.5')
        ax4.set_aspect(1)
        
        plt.savefig(F'{self.path}/sp_{tt:06d}.jpg', bbox_inches='tight', pad_inches=0.1)
        plt.cla()
        plt.clf()
        plt.close(fig)
      
    def get_kinetic_energy(self,ii):
        """
        E_k = H/2 * SUM(u2^2 + v2^2) * dx*dy
        """
        #self.E_k[ii] = 0.5 * np.sum(self.H[:,:] * (self.u2[:,:-1]**2  + self.v2[:-1,:]**2)) * self.DX * self.DY
        self.E_k[ii] = 0.5 * np.sum((self.H[:,:] * (self.u2[:,:-1]**2  + self.v2[:-1,:]**2))[1:-1, 1:-1]) * self.DX * self.DY
    
    def get_potential_energy(self,ii):
        """
        E_p = g/2 * SUM(h2^2) * dx * dy
        """
        self.E_p[ii] = 0.5 * self.gravity * np.sum((self.h2 ** 2)[1:-1, 1:-1]) * self.DX * self.DY
        
    def calc_volume(self, ii):
        self.vol[ii] = np.nansum(np.abs(self.h2)[1:-1, 1:-1]) * self.DX * self.DY
        
    def run_nesting(self):
        self.interpn_nesting()
        
        for tt in range(self.nest_ratio):
            self.u_1 = self.uxr.u1.data[... , tt] * self.umask + (1-self.umask) * self.u_1
            self.v_1 = self.vxr.v1.data[... , tt] * self.vmask + (1-self.vmask) * self.v_1
            self.h_1 = self.hxr.h1.data[... , tt] * self.hmask + (1-self.hmask) * self.h_1
            
            self.u_2 = self.uxr.u2.data[... , tt] * self.umask + (1-self.umask) * self.u_2
            self.v_2 = self.vxr.v2.data[... , tt] * self.vmask + (1-self.vmask) * self.v_2
            self.h_2 = self.hxr.h2.data[... , tt] * self.hmask + (1-self.hmask) * self.h_2
            
            self.centered_differences_nest()
            
            self.u_0 = self.u_1.copy()
            self.v_0 = self.v_1.copy()
            self.h_0 = self.h_1.copy()
            self.u_1 = self.u_2.copy()
            self.v_1 = self.v_2.copy()
            self.h_1 = self.h_2.copy()
    
    def run(self, cmap:str='viridis'):
        if self.plotting:
            self.check_dir()
        self.outputs=0
        if self.condition == 'e':
            self.create_perturbation2()
        elif self.condition == 'c':
            self.create_perturbation1()
        else:
            self.create_perturbation0()
                
        if self.use_nest:
            self.create_masks()
            self.interp_topo()
            self.xnm, self.ynm = np.meshgrid(self.xn, self.yn)
        
            
        for tt in range(self.nt):
            self.boundary_conditions()
            
            if tt == 0:
                self.forward_difference()
                
                if self.use_nest:
                    self.interp_ic()
                    
            else:
                if self.use_nest:
                    self.run_nesting()
                
                self.centered_differences()
                
                
            if self.asselin and tt%self.asselin_step==0 and tt!=0:
               self.apply_asselin()
               
            self.update_uvh(tt)
            
            if self.metrics:
                self.get_kinetic_energy(tt)
                self.get_potential_energy(tt)
                self.calc_volume(tt)
            
            if self.save_outputs and tt%self.save_interval==0:
                self.u_output[self.outputs, :] = self.u2.copy()
                self.v_output[self.outputs, :] = self.v2.copy()
                self.h_output[self.outputs, :] = self.h2.copy()
                self.outputs+=1
            
            if self.plotting and tt%self.plot_interval == 0:
                self.save_figures(tt, cmap)
                # self.plot_speed(tt)

class channel2dmodel:
    
    def __init__(self, X:int=200, DX:float=1000, 
                 Y:int=200, DY:float=1000, 
                 DT:float=3., nt:int=3000, 
                 H_0:float=2500., gravity:float=9.81, 
                 omega:float=7.29E-5, lat:float=30,
                 period:float = 1500, 
                 use_asselin:bool=False, asselin_value:float=0.1, asselin_step:int=1,
                 nesting:bool=False, nest_ratio:float=3,
                 calculate_metrics:bool=False,
                 nestpos:tuple=(150,150,30,30), # tuple(x_origin, y_origin, width, height)
                 dampening:int=10, plotting:bool=False, plot_path='sim_output', 
                 plot_interval:int=100,
                 save_outputs:bool=False, save_interval:int = 500):
        """
        X, Y    : mesh size
        DX, DY  : mesh spacing [m]
        DT      : time step [s]
        H_0     : Average depth 
        nt      : number of time steps
        period  : initial perturbation period (s)
        g       : gravity m/s^2
        condition: "c" = cosine_condition, "e" = exponential_conditon, 'D' = centered initial condition as in original code
        origin: centering for the initial condition only used if condition is exponential and cos
        size: relative size for the initial condition exponential only
        """
        self.use_nest = nesting
        self.X = X
        self.DX = DX
        self.Y = Y
        self.DY = DY
        self.DT = DT
        self.nt = nt
        self.gravity = gravity
        self.period = period
        self.H = np.ones((Y, X)) * H_0
        self.h0 = np.zeros((Y, X))
        self.h1 = np.zeros((Y, X))
        self.h2 = np.zeros((Y, X))
        self.u0 = np.zeros((Y, X+1))
        self.u1 = np.zeros((Y, X+1))
        self.u2 = np.zeros((Y, X+1))
        self.v0 = np.zeros((Y+1, X))
        self.v1 = np.zeros((Y+1, X))
        self.v2 = np.zeros((Y+1, X))
        self.CFL = np.sqrt(self.gravity*np.nanmax(self.H)) * self.DT/self.DX
        self.f = 2 * np.pi * omega * np.sin(np.deg2rad(lat))
        
        self.Xv = np.arange(self.X)
        self.Yv = np.arange(self.Y)
        self.x_p, self.y_p = np.meshgrid(self.Xv, self.Yv)
        
        # Asselin parameters
        self.asselin = use_asselin
        self.asselin_coef = asselin_value
        self.asselin_step = asselin_step
        
        self.plotting = plotting
        self.path = plot_path
        self.plot_interval = plot_interval
        
        # for metrics calculation
        self.metrics = calculate_metrics
        if self.metrics:
            self.E_k = np.zeros(nt)
            self.E_p = np.zeros(nt)
            self.vol = np.zeros(nt)
        
        # to save outputs
        self.save_outputs = save_outputs
        if self.save_outputs:
            self.save_interval = save_interval
            self.u_output = np.zeros([int(self.nt/self.save_interval)+1, self.Y, self.X+1])
            self.v_output = np.zeros([int(self.nt/self.save_interval)+1, self.Y+1, self.X])
            self.h_output = np.zeros([int(self.nt/self.save_interval)+1, self.Y, self.X])
            
        #for nested data
        if self.use_nest:
            self.nest_ratio = nest_ratio
            self.roi = nestpos
            self.dampening = dampening
            self.Xn = nestpos[2] * self.nest_ratio + 1 
            self.Yn = nestpos[3] * self.nest_ratio + 1 
            self.DTn = self.DT / self.nest_ratio
            self.DXn = self.DX / self.nest_ratio
            self.DYn = self.DY / self.nest_ratio
            self.x0nest = nestpos[0]
            self.x1nest = nestpos[0] + nestpos[2] + 1
            self.y0nest = nestpos[1]
            self.y1nest = nestpos[1] + nestpos[3] + 1
            
            self.u_1 = np.zeros([self.Yn, self.Xn + 1])
            self.u_2 = np.zeros([self.Yn, self.Xn + 1])
            self.v_1 = np.zeros([self.Yn + 1, self.Xn])
            self.v_2 = np.zeros([self.Yn + 1, self.Xn])
            self.h_1 = np.zeros([self.Yn, self.Xn])
            self.h_2 = np.zeros([self.Yn, self.Xn])
            # Parametros para la interpolacion de dominios
            self.xh_xa = np.arange(self.X)
            self.yh_xa = np.arange(self.Y)
            self.xu_xa = np.arange(-0.5,self.X+0.5)*self.DX
            self.yv_xa = np.arange(-0.5,self.Y+0.5)*self.DY
            self.t0 = np.arange(2)        
            self.xh_tg = np.linspace(self.xh_xa[self.x0nest], self.xh_xa[self.x1nest], self.Xn)
            self.yh_tg = np.linspace(self.yh_xa[self.y0nest], self.yh_xa[self.y1nest], self.Yn)
            self.xu_tg = np.linspace(self.xu_xa[self.x0nest], self.xu_xa[self.x1nest], self.Xn+1)
            self.yv_tg = np.linspace(self.yv_xa[self.y0nest], self.yv_xa[self.y1nest], self.Yn+1)
            self.tt_tg = np.linspace(0, 1, self.nest_ratio)
            
    def check_dir(self):
        if isdir(self.path) == False:
            mkdir(self.path)
    
    def create_perturbation(self, tt):
        self.h1[1,:] = 0.5 * (1 + np.cos(np.pi + 2 * np.pi * (tt - 1)/self.period))
    
    def update_uvh(self, tt):
        if tt!=0:
            self.u0 = self.u1.copy()
            self.v0 = self.v1.copy()
            self.h0 = self.h1.copy()
            self.u1 = self.u2.copy()
            self.v1 = self.v2.copy()
            self.h1 = self.h2.copy()
        else:
            self.u0 = self.u1.copy()
            self.v0 = self.v1.copy()
            self.h0 = self.h1.copy()
       
    def boundary_conditions(self, tt):
        self.u1[:,1]  = 0
        self.u1[:,-2] = 0
        self.v1[1,:]  = 0
        self.v1[-2,:] = 0
        
        if tt>self.period:
            self.h1[1] = 0
        
    def forward_difference(self):
        self.u1[:,1:-1] = self.u0[:,1:-1] -  self.gravity * self.DT/self.DX * (self.h0[:,1:] - self.h0[:,:-1]) + self.f * self.DT * (self.v1[1:,1:] + self.v1[1:,:-1] + self.v1[:-1,1:] + self.v1[:-1,:-1])/4
        self.v1[1:-1,:] = self.v0[1:-1,:] -  self.gravity * self.DT/self.DY * (self.h0[1:,:] - self.h0[:-1,:]) - self.f * self.DT * (self.u1[1:,1:] + self.u1[:-1,1:] + self.u1[1:,:-1] + self.u1[:-1,:-1])/4
        self.h1[1:-1, 1:-1] = self.h0[1:-1,1:-1] - self.H[1:-1,1:-1] * (self.DT/self.DX * (self.u0[1:-1,2:-1] - self.u0[1:-1,1:-2]) + self.DT/self.DY * (self.v0[2:-1,1:-1] - self.v0[1:-2,1:-1]))
        
        
    def centered_differences(self):
        self.u2[:,1:-1] = self.u0[:,1:-1] - 2 * self.gravity * self.DT/self.DX * (self.h1[:,1:] - self.h1[:,:-1]) + self.f * self.DT * (self.v1[1:,1:] + self.v1[1:,:-1] + self.v1[:-1,1:] + self.v1[:-1,:-1])/2
        self.v2[1:-1,:] = self.v0[1:-1,:] - 2 * self.gravity * self.DT/self.DY * (self.h1[1:,:] - self.h1[:-1,:]) - self.f * self.DT * (self.u1[1:,1:] + self.u1[:-1,1:] + self.u1[1:,:-1] + self.u1[:-1,:-1])/2
        self.h2[1:-1, 1:-1] = self.h0[1:-1,1:-1] - 2 * self.H[1:-1,1:-1] * (self.DT/self.DX * (self.u1[1:-1,2:-1] - self.u1[1:-1,1:-2]) + self.DT/self.DY * (self.v1[2:-1,1:-1] - self.v1[1:-2,1:-1]))
        
    def centered_differences_nest(self):
        self.u_2[:,1:-1] = self.u_0[:,1:-1] - 2 * self.gravity * self.DTn/self.DXn * (self.h_1[:,1:] - self.h_1[:,:-1]) + self.f * self.DTn * (self.v_1[1:,1:] + self.v_1[1:,:-1] + self.v_1[:-1,1:] + self.v_1[:-1,:-1])/4
        self.v_2[1:-1,:] = self.v_0[1:-1,:] - 2 * self.gravity * self.DTn/self.DYn * (self.h_1[1:,:] - self.h_1[:-1,:]) - self.f * self.DTn * (self.u_1[1:,1:] + self.u_1[:-1,1:] + self.u_1[1:,:-1] + self.u_1[:-1,:-1])/4
        self.h_2[1:-1, 1:-1] = self.h_0[1:-1,1:-1] - 2 * self.H2[1:-1,1:-1] * self.DTn/self.DYn * (self.u_1[1:-1,2:-1] - self.u_1[1:-1,1:-2] + self.v_1[2:-1,1:-1] - self.v_1[1:-2,1:-1])
    
    def create_masks(self):
        Is = self.dampening
        mu = 0.5 * (1+np.cos(np.pi*np.arange(0,Is + 1)/Is ))
        self.hmask = np.zeros((self.Yn, self.Xn))
        self.vmask = np.zeros((self.Yn+1, self.Xn))
        self.umask = np.zeros((self.Yn, self.Xn+1))
        
        self.hmask[:Is+1, :] = mu.reshape(Is+1,1)
        self.hmask[-Is-1:, :] = np.flipud(mu.reshape(Is+1,1))
        self.umask[:Is+1, :] = mu.reshape(Is+1,1)
        self.umask[-Is-1:, :] = np.flipud(mu.reshape(Is+1,1))
        self.vmask[:Is+1, :] = mu.reshape(Is+1,1)
        self.vmask[-Is-1:, :] = np.flipud(mu.reshape(Is+1,1))
        
        for ii in range(Is):
            for jj in range(ii+1, self.Yn-ii):
                self.hmask[jj, ii] = 0.5 * (1+np.cos(np.pi*ii/Is ))
                self.hmask[jj,-ii] = 0.5 * (1+np.cos(np.pi*ii/Is ))
                self.umask[jj, ii] = 0.5 * (1+np.cos(np.pi*ii/Is ))
                self.umask[jj,-ii] = 0.5 * (1+np.cos(np.pi*ii/Is ))
                self.vmask[jj, ii] = 0.5 * (1+np.cos(np.pi*ii/Is ))
                self.vmask[jj,-ii] = 0.5 * (1+np.cos(np.pi*ii/Is ))
    
    def interp_topo(self):
        self.H2 = xr.DataArray(self.H, dims=['y', 'x'], coords={'x':(('x'), np.arange(self.X) * self.DX),
                                                                'y':(('y'), np.arange(self.Y) * self.DY)})
        self.xn = np.linspace(self.H2.x[self.x0nest], self.H2.x[self.x1nest], self.Xn)
        self.yn = np.linspace(self.H2.y[self.y0nest], self.H2.y[self.y1nest], self.Yn)
        self.H2 = self.H2.interp(x=self.xn, y=self.yn).data
        
        
    def interp_ic(self):
        h = xr.DataArray(self.h1, dims=['y','x'], coords={'x':(('x'), self.xh_xa),
                                                          'y':(('y'), self.yh_xa)})
        u = xr.DataArray(self.u1, dims=['y','x'], coords={'x':(('x'), self.xu_xa),
                                                          'y':(('y'), self.yh_xa)})
        v = xr.DataArray(self.v1, dims=['y','x'], coords={'x':(('x'), self.xh_xa),
                                                          'y':(('y'), self.yv_xa)})
        self.u_0 = u.interp(x=self.xu_tg, y=self.yh_tg).data.copy()
        self.v_0 = v.interp(x=self.xh_tg, y=self.yv_tg).data.copy()
        self.h_0 = h.interp(x=self.xh_tg, y=self.yh_tg).data.copy()
    
    def interpn_nesting(self):
        h1 = np.stack([self.h0, self.h1], axis=-1)
        h2 = np.stack([self.h1, self.h2], axis=-1)
        u1 = np.stack([self.u0, self.u1], axis=-1)
        u2 = np.stack([self.u1, self.u2], axis=-1)
        v1 = np.stack([self.v0, self.v1], axis=-1)
        v2 = np.stack([self.v1, self.v2], axis=-1)
        
        self.hxr = xr.Dataset(
            data_vars=dict(h1=(('y','x','t'), h1), h2=(('y','x','t'), h2)),
            coords={'x':(('x'), self.xh_xa), 
                    'y':(('y'), self.yh_xa),
                    't':(('t'), self.t0)})
        self.uxr = xr.Dataset(
            data_vars=dict(u1=(('y','x','t'), u1), u2=(('y','x','t'), u2)),
            coords={'x':(('x'), self.xu_xa), 
                    'y':(('y'), self.yh_xa),
                    't':(('t'), self.t0)})
        self.vxr = xr.Dataset(
            data_vars=dict(v1=(('y','x','t'), v1), v2=(('y','x','t'), v2)),
            coords={'x':(('x'), self.xh_xa), 
                    'y':(('y'), self.yv_xa),
                    't':(('t'), self.t0)})
               
        self.hxr = self.hxr.interp(x=self.xh_tg, y=self.yh_tg, t=self.tt_tg)
        self.uxr = self.uxr.interp(x=self.xu_tg, y=self.yh_tg, t=self.tt_tg)
        self.vxr = self.vxr.interp(x=self.xh_tg, y=self.yv_tg, t=self.tt_tg)
        
    def run_nesting(self):
        self.interpn_nesting()
        
        for tt in range(self.nest_ratio):
            self.u_1 = self.uxr.u1.data[... , tt] * self.umask + (1-self.umask) * self.u_1
            self.v_1 = self.vxr.v1.data[... , tt] * self.vmask + (1-self.vmask) * self.v_1
            self.h_1 = self.hxr.h1.data[... , tt] * self.hmask + (1-self.hmask) * self.h_1
            
            self.u_2 = self.uxr.u2.data[... , tt] * self.umask + (1-self.umask) * self.u_2
            self.v_2 = self.vxr.v2.data[... , tt] * self.vmask + (1-self.vmask) * self.v_2
            self.h_2 = self.hxr.h2.data[... , tt] * self.hmask + (1-self.hmask) * self.h_2
            
            self.centered_differences_nest()
            
            self.u_0 = self.u_1.copy()
            self.v_0 = self.v_1.copy()
            self.h_0 = self.h_1.copy()
            self.u_1 = self.u_2.copy()
            self.v_1 = self.v_2.copy()
            self.h_1 = self.h_2.copy()
    
    def apply_asselin(self):
        self.u1 += self.asselin_coef * (self.u0 - 2*self.u1 + self.u2)
        self.v1 += self.asselin_coef * (self.v0 - 2*self.v1 + self.v2)
        self.h1 += self.asselin_coef * (self.h0 - 2*self.h1 + self.h2)
        
    def get_kinetic_energy(self,ii):
        """
        E_k = H/2 * SUM(u2^2 + v2^2) * dx * dy
        """
        self.E_k[ii] = 0.5 * np.sum((self.H[:,:] * (self.u2[:,:-1]**2  + self.v2[:-1,:]**2))[1:-1, 1:-1]) * self.DX * self.DY
    
    def get_potential_energy(self,ii):
        """
        E_p = g/2 * SUM(h2^2) * dx * dy
        """
        self.E_p[ii] = 0.5 * self.gravity * np.sum((self.h2 ** 2)[1:-1, 1:-1]) * self.DX * self.DY
        
    def calc_volume(self, ii):
        self.vol[ii] = np.nansum(np.abs(self.h2)[1:-1, 1:-1]) * self.DX * self.DY
    
    def plot_speed(self, tt):
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,6), dpi=300)
        fig.suptitle(F'CFL:{self.CFL:06f}, timestep:{tt}', fontsize=16)
        pc1 = ax1.pcolormesh(self.h2.T, vmin=-1, vmax=1, cmap='twilight_shifted')
        plt.colorbar(pc1, ax=ax1,label='h (m)')
        ax1.set_ylabel('x')
        ax1.axhline(100,linestyle='--', color='g', linewidth=0.8)
        ax1.axhline(190,linestyle='--', color='r', linewidth=0.8)
        ax1.axhline(10,linestyle='--', color='b', linewidth=0.8)
        ax1.set_aspect(1)

        ax2.plot(self.h2[:,100], 'g-')
        ax2.plot(self.h2[:,190], 'r-')
        ax2.plot(self.h2[:,10], 'b-')
        ax2.set_ylabel('h (m)')
        ax2.set_ylim(-1.5,1.5)
        ax2.grid()

        pc3 = ax3.pcolormesh(self.u2.T, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        plt.colorbar(pc3, ax=ax3,label='u (m/s)')
        ax3.set_xlabel('y')
        ax3.axhline(100,linestyle='--', color='0.5')
        ax3.set_aspect(1)

        pc4 = ax4.pcolormesh(self.v2.T, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        plt.colorbar(pc4, ax=ax4,label='v (m/s)')
        ax4.set_xlabel('y')
        ax4.set_ylabel('x')
        ax4.axhline(100,linestyle='--', color='0.5')
        ax4.set_aspect(1)
        
        plt.savefig(F'{self.path}/sp_{tt:06d}.jpg', bbox_inches='tight', pad_inches=0.1)
        plt.cla()
        plt.clf()
        plt.close(fig)
    
    def save_figures(self, tt, cmap='viridis'):
        fig = plt.figure(figsize=(13,6), dpi=300)
        fig.suptitle(F'CFL:{self.CFL:06f}, timestep:{tt}', fontsize=16)
        if self.use_nest:
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3, projection='3d')
            ax4 = fig.add_subplot(2, 2, 4)
            roi = patches.Rectangle((self.roi[1], self.roi[0]), self.roi[3], self.roi[2], 
                                    linewidth=1, edgecolor='r', facecolor='none', zorder=100)
            ax2.add_patch(roi)
            
            ax3.set_xlabel('y (km)')
            ax3.set_ylabel('x (km)')
            ax3.set_zlabel('z (m)')
            ax4.set_xlabel('y (km)')
            ax4.set_ylabel('x (km)')
        else:
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2)
            
            
        if tt == 0:
            ax1.plot_surface(self.y_p[1:-1,1:-1].T, self.x_p[1:-1,1:-1].T, self.h1[1:-1,1:-1].T, cmap=cmap, edgecolor='none', antialiased=True, vmin=-1, vmax=1)
            img = ax2.pcolormesh(self.y_p.T, self.x_p.T,self.h1.T, vmin=-1, vmax=1, cmap=cmap)
            if self.use_nest:
                ax4.pcolormesh(self.ynm.T/1000, self.xnm.T/1000, self.h_1.T, vmin=-1, vmax=1, cmap=cmap)
                ax3.plot_surface(self.xnm/1000, self.ynm/1000, self.h_1, cmap=cmap, edgecolor='none', antialiased=True, vmin=-1, vmax=1)
                ax4.set_aspect(1)
        else:
            ax1.plot_surface(self.y_p[1:-1,1:-1].T, self.x_p[1:-1,1:-1].T, self.h2[1:-1,1:-1].T, cmap=cmap, edgecolor='none', antialiased=True, vmin=-1, vmax=1)
            img = ax2.pcolormesh(self.h2.T, vmin=-1, vmax=1, cmap=cmap)
            if self.use_nest:
                ax4.pcolormesh(self.ynm.T/1000, self.xnm.T/1000, self.h_2.T, vmin=-1, vmax=1, cmap=cmap)
                ax3.plot_surface(self.ynm.T/1000, self.xnm.T/1000, self.h_2.T, cmap=cmap, edgecolor='none', antialiased=True, vmin=-1, vmax=1)
                ax4.set_aspect(1)
                         
        ax2.set_aspect(1)        
        ax1.set_xlabel('y (km)')
        ax1.set_ylabel('x (km)')
        ax1.set_zlabel('z (m)')
        ax2.set_xlabel('y (km)')
        ax2.set_ylabel('x (km)')
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
        fig.colorbar(img, cax=cbar_ax, label='$h$ (m)')
        plt.savefig(F'{self.path}/{tt:06d}.jpg', bbox_inches='tight', pad_inches=0.1)
        plt.cla()
        plt.clf()
        plt.close(fig)
        
       
    def run(self, cmap:str='viridis'):
        if self.plotting:
            self.check_dir()
            
        self.outputs=0
        
        if self.use_nest:
            self.create_masks()
            self.interp_topo()
            self.xnm, self.ynm = np.meshgrid(self.xn, self.yn)
        
            
        for tt in range(self.nt):
            if tt <= self.period:
                self.create_perturbation(tt)
            
            self.boundary_conditions(tt)
            
            if tt == 0:                
                self.forward_difference()
                
                if self.use_nest:
                    self.interp_ic()
                    
            else:
                if self.use_nest:
                    self.run_nesting()
                
                self.centered_differences()
                
            if self.asselin and tt%self.asselin_step==0 and tt!=0:
                self.apply_asselin()
                
            self.update_uvh(tt)
            
            if self.metrics:
                self.get_kinetic_energy(tt)
                self.get_potential_energy(tt)
                self.calc_volume(tt)
            
            if self.save_outputs and tt%self.save_interval==0:
                self.u_output[self.outputs, :] = self.u2.copy()
                self.v_output[self.outputs, :] = self.v2.copy()
                self.h_output[self.outputs, :] = self.h2.copy()
                self.outputs+=1
            
            if self.plotting and tt%self.plot_interval == 0:
                self.save_figures(tt, cmap)
                # self.plot_speed(tt)
