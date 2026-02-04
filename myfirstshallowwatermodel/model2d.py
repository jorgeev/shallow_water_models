# -*- coding: utf-8 -*-
"""
Shallow water model in 2D with support for nesting inside the domain.
version 2025.02.03T19
author jorgeev@github.com

Refactored: base class ShallowWaterModel2DBase, concrete models simple2dmodel,
channel2dmodel, and custom_ic_2dmodel. Access all via ShallowWaterModel factory.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits import mplot3d
from os import mkdir
from os.path import isdir
from typing import Tuple, Union

# Default Coriolis: f = 2*omega*sin(lat), omega in rad/s
_DEFAULT_OMEGA = 7.29e-5


def _ensure_shape(arr: np.ndarray, shape: Tuple[int, ...], name: str) -> np.ndarray:
    """Copy and ensure array has exact shape; broadcast or crop if needed."""
    arr = np.asarray(arr, dtype=float)
    if arr.shape != shape:
        if arr.size == 1:
            return np.full(shape, float(arr.flat[0]))
        # Try to broadcast or crop to shape
        out = np.zeros(shape, dtype=float)
        sy, sx = min(arr.shape[0], shape[0]), min(arr.shape[1], shape[1])
        out[:sy, :sx] = arr[:sy, :sx]
        return out
    return arr.copy()


def _depth_to_H(H_0: Union[float, np.ndarray], Y: int, X: int) -> np.ndarray:
    """
    Convert depth specification to grid depth array self.H.
    - If H_0 is a scalar (float/int): return (Y, X) array with constant value.
    - If H_0 is an array: must have shape (Y, X); return a copy (cropped/padded to (Y, X) if needed).
    """
    if np.isscalar(H_0):
        return np.full((Y, X), float(H_0), dtype=float)
    arr = np.asarray(H_0, dtype=float)
    if arr.shape != (Y, X):
        out = np.zeros((Y, X), dtype=float)
        sy, sx = min(arr.shape[0], Y), min(arr.shape[1], X)
        out[:sy, :sx] = arr[:sy, :sx]
        return out
    return arr.copy()


class ShallowWaterModel2DBase:
    """
    Base class for 2D shallow water models. Holds mesh, physics (H, g, f),
    time-stepping arrays (h0/h1/h2, u0/u1/u2, v0/v1/v2), nesting, outputs, and metrics.
    Subclasses implement: set_initial_conditions(), boundary_conditions(tt), and optionally run_pre_step(tt).
    H_0: depth (m); float = constant depth, or (Y,X) array = depth at each grid point → becomes self.H.
    """

    def __init__(
        self,
        X: int = 200,
        DX: float = 1000,
        Y: int = 200,
        DY: float = 1000,
        DT: float = 3.0,
        nt: int = 3000,
        H_0: Union[float, np.ndarray] = 1500.0,
        gravity: float = 9.81,
        omega: float = _DEFAULT_OMEGA,
        lat: float = 30,
        nesting: bool = False,
        nest_ratio: float = 3,
        use_asselin: bool = False,
        asselin_value: float = 0.1,
        asselin_step: int = 1,
        calculate_metrics: bool = False,
        nestpos: Tuple[int, int, int, int] = (150, 150, 30, 30),
        dampening: int = 10,
        plotting: bool = False,
        plot_path: str = "sim_output",
        plot_interval: int = 100,
        exp_name: str = "",
        save_outputs: bool = False,
        save_interval: int = 500,
    ):
        self.use_nest = nesting
        self.X, self.DX = X, DX
        self.Y, self.DY = Y, DY
        self.DT, self.nt = DT, nt
        self.gravity = gravity
        self.H = _depth_to_H(H_0, Y, X)
        self.f = 2 * omega * np.sin(np.deg2rad(lat))

        self.h0 = np.zeros((Y, X))
        self.h1 = np.zeros((Y, X))
        self.h2 = np.zeros((Y, X))
        self.u0 = np.zeros((Y, X + 1))
        self.u1 = np.zeros((Y, X + 1))
        self.u2 = np.zeros((Y, X + 1))
        self.v0 = np.zeros((Y + 1, X))
        self.v1 = np.zeros((Y + 1, X))
        self.v2 = np.zeros((Y + 1, X))

        self.CFL = np.sqrt(self.gravity * np.nanmax(self.H)) * self.DT / self.DX
        self.Xv = np.arange(self.X)
        self.Yv = np.arange(self.Y)
        self.x_p, self.y_p = np.meshgrid(self.Xv, self.Yv)

        self.plotting = plotting
        self.plot_path = plot_path
        self.exp_name = (exp_name or "").strip()
        self.path = f"{plot_path}/{self.exp_name}" if self.exp_name else plot_path
        self.plot_interval = plot_interval
        self.asselin = use_asselin
        self.asselin_coef = asselin_value
        self.asselin_step = asselin_step
        self.metrics = calculate_metrics
        if self.metrics:
            self.E_k = np.zeros(nt)
            self.E_p = np.zeros(nt)
            self.vol = np.zeros(nt)

        self.save_outputs = save_outputs
        if self.save_outputs:
            self.save_interval = save_interval
            self.u_output = np.zeros([int(self.nt / self.save_interval) + 1, self.Y, self.X + 1])
            self.v_output = np.zeros([int(self.nt / self.save_interval) + 1, self.Y + 1, self.X])
            self.h_output = np.zeros([int(self.nt / self.save_interval) + 1, self.Y, self.X])

        if self.use_nest:
            self.nest_ratio = nest_ratio
            self.roi = nestpos
            self.dampening = dampening
            self.Xn = nestpos[2] * int(nest_ratio) + 1
            self.Yn = nestpos[3] * int(nest_ratio) + 1
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
            self.xh_xa = np.arange(self.X)
            self.yh_xa = np.arange(self.Y)
            self.xu_xa = np.arange(-0.5, self.X + 0.5) * self.DX
            self.yv_xa = np.arange(-0.5, self.Y + 0.5) * self.DY
            self.t0 = np.arange(2)
            self.xh_tg = np.linspace(self.xh_xa[self.x0nest], self.xh_xa[self.x1nest], self.Xn)
            self.yh_tg = np.linspace(self.yh_xa[self.y0nest], self.yh_xa[self.y1nest], self.Yn)
            self.xu_tg = np.linspace(self.xu_xa[self.x0nest], self.xu_xa[self.x1nest], self.Xn + 1)
            self.yv_tg = np.linspace(self.yv_xa[self.y0nest], self.yv_xa[self.y1nest], self.Yn + 1)
            self.tt_tg = np.linspace(0, 1, int(self.nest_ratio))

    def check_dir(self):
        if not isdir(self.plot_path):
            mkdir(self.plot_path)
        if self.exp_name and not isdir(self.path):
            mkdir(self.path)

    def set_initial_conditions(self) -> None:
        """Set h0, u0, v0 (and optionally h1) before first step. Override in subclass."""
        pass

    def run_pre_step(self, tt: int) -> None:
        """Called at start of each time step (e.g. time-dependent forcing). Default: no-op."""
        pass

    def boundary_conditions(self, tt: int = 0) -> None:
        """Apply boundary conditions to u1, v1, h1. Override in subclass."""
        self.u1[:, 1] = 0
        self.u1[:, -2] = 0
        self.v1[1, :] = 0
        self.v1[-2, :] = 0

    def forward_difference(self) -> None:
        self.u1[:, 1:-1] = (
            self.u0[:, 1:-1]
            - self.gravity * self.DT / self.DX * (self.h0[:, 1:] - self.h0[:, :-1])
            + self.f * self.DT * (self.v1[1:, 1:] + self.v1[1:, :-1] + self.v1[:-1, 1:] + self.v1[:-1, :-1]) / 4
        )
        self.v1[1:-1, :] = (
            self.v0[1:-1, :]
            - self.gravity * self.DT / self.DY * (self.h0[1:, :] - self.h0[:-1, :])
            - self.f * self.DT * (self.u1[1:, 1:] + self.u1[:-1, 1:] + self.u1[1:, :-1] + self.u1[:-1, :-1]) / 4
        )
        self.h1[1:-1, 1:-1] = self.h0[1:-1, 1:-1] - self.H[1:-1, 1:-1] * (
            self.DT / self.DX * (self.u0[1:-1, 2:-1] - self.u0[1:-1, 1:-2])
            + self.DT / self.DY * (self.v0[2:-1, 1:-1] - self.v0[1:-2, 1:-1])
        )

    def centered_differences(self) -> None:
        self.u2[:, 1:-1] = (
            self.u0[:, 1:-1]
            - 2 * self.gravity * self.DT / self.DX * (self.h1[:, 1:] - self.h1[:, :-1])
            + self.f * self.DT * (self.v1[1:, 1:] + self.v1[1:, :-1] + self.v1[:-1, 1:] + self.v1[:-1, :-1]) / 2
        )
        self.v2[1:-1, :] = (
            self.v0[1:-1, :]
            - 2 * self.gravity * self.DT / self.DY * (self.h1[1:, :] - self.h1[:-1, :])
            - self.f * self.DT * (self.u1[1:, 1:] + self.u1[:-1, 1:] + self.u1[1:, :-1] + self.u1[:-1, :-1]) / 2
        )
        self.h2[1:-1, 1:-1] = self.h0[1:-1, 1:-1] - 2 * self.H[1:-1, 1:-1] * (
            self.DT / self.DX * (self.u1[1:-1, 2:-1] - self.u1[1:-1, 1:-2])
            + self.DT / self.DY * (self.v1[2:-1, 1:-1] - self.v1[1:-2, 1:-1])
        )

    def centered_differences_nest(self) -> None:
        self.u_2[:, 1:-1] = (
            self.u_0[:, 1:-1]
            - 2 * self.gravity * self.DTn / self.DXn * (self.h_1[:, 1:] - self.h_1[:, :-1])
            + self.f * self.DTn * (self.v_1[1:, 1:] + self.v_1[1:, :-1] + self.v_1[:-1, 1:] + self.v_1[:-1, :-1]) / 4
        )
        self.v_2[1:-1, :] = (
            self.v_0[1:-1, :]
            - 2 * self.gravity * self.DTn / self.DYn * (self.h_1[1:, :] - self.h_1[:-1, :])
            - self.f * self.DTn * (self.u_1[1:, 1:] + self.u_1[:-1, 1:] + self.u_1[1:, :-1] + self.u_1[:-1, :-1]) / 4
        )
        self.h_2[1:-1, 1:-1] = self.h_0[1:-1, 1:-1] - 2 * self.H2[1:-1, 1:-1] * self.DTn / self.DYn * (
            self.u_1[1:-1, 2:-1] - self.u_1[1:-1, 1:-2] + self.v_1[2:-1, 1:-1] - self.v_1[1:-2, 1:-1]
        )

    def create_masks(self) -> None:
        Is = self.dampening
        mu = 0.5 * (1 + np.cos(np.pi * np.arange(0, Is + 1) / Is))
        self.hmask = np.zeros((self.Yn, self.Xn))
        self.vmask = np.zeros((self.Yn + 1, self.Xn))
        self.umask = np.zeros((self.Yn, self.Xn + 1))
        self.hmask[: Is + 1, :] = mu.reshape(Is + 1, 1)
        self.hmask[-Is - 1 :, :] = np.flipud(mu.reshape(Is + 1, 1))
        self.umask[: Is + 1, :] = mu.reshape(Is + 1, 1)
        self.umask[-Is - 1 :, :] = np.flipud(mu.reshape(Is + 1, 1))
        self.vmask[: Is + 1, :] = mu.reshape(Is + 1, 1)
        self.vmask[-Is - 1 :, :] = np.flipud(mu.reshape(Is + 1, 1))
        for ii in range(Is):
            for jj in range(ii + 1, self.Yn - ii):
                self.hmask[jj, ii] = 0.5 * (1 + np.cos(np.pi * ii / Is))
                self.hmask[jj, -ii] = 0.5 * (1 + np.cos(np.pi * ii / Is))
                self.umask[jj, ii] = 0.5 * (1 + np.cos(np.pi * ii / Is))
                self.umask[jj, -ii] = 0.5 * (1 + np.cos(np.pi * ii / Is))
                self.vmask[jj, ii] = 0.5 * (1 + np.cos(np.pi * ii / Is))
                self.vmask[jj, -ii] = 0.5 * (1 + np.cos(np.pi * ii / Is))

    def interp_topo(self) -> None:
        self.H2 = xr.DataArray(
            self.H,
            dims=["y", "x"],
            coords={"x": (("x"), np.arange(self.X) * self.DX), "y": (("y"), np.arange(self.Y) * self.DY)},
        )
        self.xn = np.linspace(self.H2.x[self.x0nest].values, self.H2.x[self.x1nest].values, self.Xn)
        self.yn = np.linspace(self.H2.y[self.y0nest].values, self.H2.y[self.y1nest].values, self.Yn)
        self.H2 = self.H2.interp(x=self.xn, y=self.yn)

    def interp_ic(self) -> None:
        h = xr.DataArray(self.h1, dims=["y", "x"], coords={"x": (("x"), self.xh_xa), "y": (("y"), self.yh_xa)})
        u = xr.DataArray(self.u1, dims=["y", "x"], coords={"x": (("x"), self.xu_xa), "y": (("y"), self.yh_xa)})
        v = xr.DataArray(self.v1, dims=["y", "x"], coords={"x": (("x"), self.xh_xa), "y": (("y"), self.yv_xa)})
        self.u_0 = u.interp(x=self.xu_tg, y=self.yh_tg).data.copy()
        self.v_0 = v.interp(x=self.xh_tg, y=self.yv_tg).data.copy()
        self.h_0 = h.interp(x=self.xh_tg, y=self.yh_tg).data.copy()

    def interpn_nesting(self) -> None:
        h1 = np.stack([self.h0, self.h1], axis=-1)
        h2 = np.stack([self.h1, self.h2], axis=-1)
        u1 = np.stack([self.u0, self.u1], axis=-1)
        u2 = np.stack([self.u1, self.u2], axis=-1)
        v1 = np.stack([self.v0, self.v1], axis=-1)
        v2 = np.stack([self.v1, self.v2], axis=-1)
        self.hxr = xr.Dataset(
            data_vars=dict(h1=(("y", "x", "t"), h1), h2=(("y", "x", "t"), h2)),
            coords={"x": (("x"), self.xh_xa), "y": (("y"), self.yh_xa), "t": (("t"), self.t0)},
        )
        self.uxr = xr.Dataset(
            data_vars=dict(u1=(("y", "x", "t"), u1), u2=(("y", "x", "t"), u2)),
            coords={"x": (("x"), self.xu_xa), "y": (("y"), self.yh_xa), "t": (("t"), self.t0)},
        )
        self.vxr = xr.Dataset(
            data_vars=dict(v1=(("y", "x", "t"), v1), v2=(("y", "x", "t"), v2)),
            coords={"x": (("x"), self.xh_xa), "y": (("y"), self.yv_xa), "t": (("t"), self.t0)},
        )
        self.hxr = self.hxr.interp(x=self.xh_tg, y=self.yh_tg, t=self.tt_tg)
        self.uxr = self.uxr.interp(x=self.xu_tg, y=self.yh_tg, t=self.tt_tg)
        self.vxr = self.vxr.interp(x=self.xh_tg, y=self.yv_tg, t=self.tt_tg)

    def run_nesting(self) -> None:
        self.interpn_nesting()
        for tt in range(int(self.nest_ratio)):
            self.u_1 = self.uxr.u1.data[..., tt] * self.umask + (1 - self.umask) * self.u_1
            self.v_1 = self.vxr.v1.data[..., tt] * self.vmask + (1 - self.vmask) * self.v_1
            self.h_1 = self.hxr.h1.data[..., tt] * self.hmask + (1 - self.hmask) * self.h_1
            self.u_2 = self.uxr.u2.data[..., tt] * self.umask + (1 - self.umask) * self.u_2
            self.v_2 = self.vxr.v2.data[..., tt] * self.vmask + (1 - self.vmask) * self.v_2
            self.h_2 = self.hxr.h2.data[..., tt] * self.hmask + (1 - self.hmask) * self.h_2
            self.centered_differences_nest()
            self.u_0 = self.u_1.copy()
            self.v_0 = self.v_1.copy()
            self.h_0 = self.h_1.copy()
            self.u_1 = self.u_2.copy()
            self.v_1 = self.v_2.copy()
            self.h_1 = self.h_2.copy()

    def apply_asselin(self) -> None:
        self.u1 += self.asselin_coef * (self.u0 - 2 * self.u1 + self.u2)
        self.v1 += self.asselin_coef * (self.v0 - 2 * self.v1 + self.v2)
        self.h1 += self.asselin_coef * (self.h0 - 2 * self.h1 + self.h2)

    def update_uvh(self, tt: int) -> None:
        if tt != 0:
            self.u0, self.v0, self.h0 = self.u1.copy(), self.v1.copy(), self.h1.copy()
            self.u1, self.v1, self.h1 = self.u2.copy(), self.v2.copy(), self.h2.copy()
        else:
            self.u0, self.v0, self.h0 = self.u1.copy(), self.v1.copy(), self.h1.copy()

        # --- momentum update ---
        #self.u1[:, 1:-1] = self.u0[:, 1:-1] - self.DT * self.gravity * (self.h0[:, 1:] - self.h0[:, :-1]) / self.DX
        #self.v1[1:-1, :] = self.v0[1:-1, :] - self.DT * self.gravity * (self.h0[1:, :] - self.h0[:-1, :]) / self.DY

        # --- enforce closed (impermeable) boundaries ---
        # zonal velocity (normal to west/east walls)
        self.u1[:, 0]  = 0.0
        self.u1[:, -1] = 0.0
        self.u0[:, 0]  = self.u0[:, -1] = 0.0
        self.v0[0, :]  = self.v0[-1, :] = 0.0
        # meridional velocity (normal to south/north walls)
        self.v1[0, :]  = 0.0
        self.v1[-1, :] = 0.0

    def get_kinetic_energy(self, ii: int) -> None:
        self.E_k[ii] = (
            0.5
            * np.sum((self.H * (self.u2[:, :-1] ** 2 + self.v2[:-1, :] ** 2))[1:-1, 1:-1])
            * self.DX
            * self.DY
        )

    def get_potential_energy(self, ii: int) -> None:
        self.E_p[ii] = 0.5 * self.gravity * np.sum((self.h2 ** 2)[1:-1, 1:-1]) * self.DX * self.DY

    def calc_volume(self, ii: int) -> None:
        self.vol[ii] = np.nansum(np.abs(self.h2)[1:-1, 1:-1]) * self.DX * self.DY

    def save_figures(self, tt: int, cmap: str = "viridis") -> None:
        """Default 2D/3D figure layout. Subclasses can override for different layouts."""
        fig = plt.figure(figsize=(13, 6), dpi=300)
        fig.suptitle(f"CFL:{self.CFL:06f}, timestep:{tt}", fontsize=16)
        if self.use_nest:
            ax1 = fig.add_subplot(2, 2, 1, projection="3d")
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3, projection="3d")
            ax4 = fig.add_subplot(2, 2, 4)
            roi = patches.Rectangle(
                (self.roi[0], self.roi[1]), self.roi[2], self.roi[3],
                linewidth=1, edgecolor="r", facecolor="none", zorder=100,
            )
            ax2.add_patch(roi)
            for ax in (ax3, ax4):
                ax.set_xlabel("y (km)")
                ax.set_ylabel("x (km)")
            ax3.set_zlabel("z (m)")
        else:
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2 = fig.add_subplot(1, 2, 2)
        h_plot = self.h1 if tt == 0 else self.h2
        vmin, vmax = -0.5, 0.5
        ax1.plot_surface(
            self.x_p[1:-1, 1:-1], self.y_p[1:-1, 1:-1], h_plot[1:-1, 1:-1],
            cmap=cmap, edgecolor="none", antialiased=True, vmin=vmin, vmax=vmax,
        )
        img = ax2.pcolormesh(self.x_p, self.y_p, h_plot, vmin=vmin, vmax=vmax, cmap=cmap)
        if self.use_nest:
            h_n = self.h_1 if tt == 0 else self.h_2
            if not hasattr(self, "xnm"):
                self.xnm, self.ynm = np.meshgrid(self.xn, self.yn)
            ax4.pcolormesh(self.xnm / 1000, self.ynm / 1000, h_n, vmin=vmin, vmax=vmax, cmap=cmap)
            ax3.plot_surface(
                self.xnm / 1000, self.ynm / 1000, h_n,
                cmap=cmap, edgecolor="none", antialiased=True, vmin=vmin, vmax=vmax,
            )
            ax4.set_aspect(1)
        ax2.set_aspect(1)
        ax1.set_xlabel("y (km)")
        ax1.set_ylabel("x (km)")
        ax1.set_zlabel("z (m)")
        ax2.set_xlabel("y (km)")
        ax2.set_ylabel("x (km)")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
        fig.colorbar(img, cax=cbar_ax, label="$h$ (m)")
        plt.savefig(f"{self.path}/{tt:06d}.jpg", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    def run(self, cmap: str = "twilight_shifted") -> None:
        if self.plotting:
            self.check_dir()
        self.outputs = 0
        self.set_initial_conditions()
        if self.use_nest:
            self.create_masks()
            self.interp_topo()
            self.xnm, self.ynm = np.meshgrid(self.xn, self.yn)
        for tt in range(self.nt):
            self.run_pre_step(tt)
            self.boundary_conditions(tt)
            if tt == 0:
                self.forward_difference()
                if self.use_nest:
                    self.interp_ic()
            else:
                if self.use_nest:
                    self.run_nesting()
                self.centered_differences()
            if self.asselin and tt % self.asselin_step == 0 and tt != 0:
                self.apply_asselin()
            self.update_uvh(tt)
            if self.metrics:
                self.get_kinetic_energy(tt)
                self.get_potential_energy(tt)
                self.calc_volume(tt)
            if self.save_outputs and tt % self.save_interval == 0:
                self.u_output[self.outputs, :] = self.u2.copy()
                self.v_output[self.outputs, :] = self.v2.copy()
                self.h_output[self.outputs, :] = self.h2.copy()
                self.outputs += 1
            if self.plotting and tt % self.plot_interval == 0:
                self.save_figures(tt, cmap)
                print(f"  Saved frame {tt:06d}.jpg")


# ---------------------------------------------------------------------------
# Simple 2D model: built-in initial conditions (cosine, exponential, centered)
# ---------------------------------------------------------------------------

class simple2dmodel(ShallowWaterModel2DBase):
    """
    Simple 2D shallow water model with built-in initial conditions:
    'c' = cosine bump at origin, 'e' = exponential, else centered bump.
    """

    def __init__(
        self,
        X: int = 200,
        DX: float = 1000,
        Y: int = 200,
        DY: float = 1000,
        DT: float = 3.0,
        nt: int = 3000,
        H_0: float = 1500.0,
        gravity: float = 9.81,
        initialc: str = "c",
        omega: float = _DEFAULT_OMEGA,
        lat: float = 30,
        period: float = 5000,
        nesting: bool = False,
        nest_ratio: float = 3,
        use_asselin: bool = False,
        asselin_value: float = 0.1,
        asselin_step: int = 1,
        calculate_metrics: bool = False,
        nestpos: Tuple[int, int, int, int] = (150, 150, 30, 30),
        dampening: int = 10,
        plotting: bool = False,
        plot_path: str = "sim_output",
        plot_interval: int = 100,
        exp_name: str = "",
        origin: Tuple[int, int] = (100, 100),
        size: Tuple[float, float] = (2, 2),
        maxh0: float = 1.0,
        save_outputs: bool = False,
        save_interval: int = 500,
        **kwargs,
    ):
        super().__init__(
            X=X, DX=DX, Y=Y, DY=DY, DT=DT, nt=nt, H_0=H_0, gravity=gravity,
            omega=omega, lat=lat, nesting=nesting, nest_ratio=nest_ratio,
            use_asselin=use_asselin, asselin_value=asselin_value, asselin_step=asselin_step,
            calculate_metrics=calculate_metrics, nestpos=nestpos, dampening=dampening,
            plotting=plotting, plot_path=plot_path, plot_interval=plot_interval,
            exp_name=exp_name, save_outputs=save_outputs, save_interval=save_interval, **kwargs,
        )
        self.condition = initialc
        self.originX, self.originY = origin[0], origin[1]
        self.size = size
        self.B = maxh0

    def set_initial_conditions(self) -> None:
        if self.condition == "e":
            self._create_perturbation_exp()
        elif self.condition == "c":
            self._create_perturbation_cosine()
        else:
            self._create_perturbation_centered()

    def _create_perturbation_centered(self) -> None:
        z = self.h0.copy()
        for jj in range(int(self.Y / 2 - 0.2 * (self.Y / 2)), int(self.Y / 2 + 0.2 * (self.Y / 2) + 1)):
            for ii in range(int(self.X / 2 - 0.2 * (self.X / 2)), int(self.X / 2 + 0.2 * (self.X / 2) + 1)):
                r = np.sqrt((ii - np.round(self.X / 2)) ** 2 + (jj - np.round(self.Y / 2)) ** 2)
                if r <= 0.2 * (self.X / 2):
                    z[jj, ii] = 0.5 * (1 + np.cos(np.pi * r / (0.2 * (self.X / 2))))
        self.h0 = z

    def _create_perturbation_cosine(self) -> None:
        z = self.h0.copy()
        for jj in range(int(self.originY - 0.2 * self.originY), int(self.originY + 0.2 * self.originY + 1)):
            for ii in range(int(self.originX - 0.2 * self.originX), int(self.originX + 0.2 * self.originX + 1)):
                r = np.sqrt((ii - np.round(self.originX)) ** 2 + (jj - np.round(self.originY)) ** 2)
                if r <= 0.2 * self.originX and r <= 0.2 * self.originY:
                    z[jj, ii] = 0.5 * (1 + np.cos(np.pi * r / (0.2 * (self.X / 2))))
        self.h0 = z

    def _create_perturbation_exp(self) -> None:
        sigma_x = 0.2 * self.X / self.size[0]
        sigma_y = 0.2 * self.X / self.size[1]
        z = self.B * np.exp(
            -((self.x_p - self.originX) ** 2 / (2 * sigma_x ** 2) + (self.y_p - self.originY) ** 2 / (2 * sigma_y ** 2))
        )
        self.h0 = np.nan_to_num(z)

    def plot_speed(self, tt: int) -> None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6), dpi=300)
        ax1.pcolormesh(self.h2, vmin=-0.5, vmax=0.5, cmap="PRGn")
        ax1.set_ylabel("y")
        ax1.axhline(100, linestyle="--", color="0.5")
        ax1.set_aspect(1)
        ax2.plot(self.h2[150, :], "g-")
        ax2.plot(self.u2[150, :], "r-")
        ax2.plot(self.v2[150, :], "b-")
        ax2.set_ylabel("h (m)")
        ax3.pcolormesh(self.u2, cmap="RdBu")
        ax3.set_xlabel("x")
        ax3.axhline(100, linestyle="--", color="0.5")
        ax3.set_aspect(1)
        ax4.pcolormesh(self.v2, cmap="RdBu")
        ax4.set_xlabel("x")
        ax4.set_ylabel("y")
        ax4.axhline(100, linestyle="--", color="0.5")
        ax4.set_aspect(1)
        plt.savefig(f"{self.path}/sp_{tt:06d}.jpg", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Channel 2D model: time-dependent boundary perturbation
# ---------------------------------------------------------------------------

class channel2dmodel(ShallowWaterModel2DBase):
    """Channel model with time-dependent h perturbation at one boundary."""

    def __init__(
        self,
        X: int = 200,
        DX: float = 1000,
        Y: int = 200,
        DY: float = 1000,
        DT: float = 3.0,
        nt: int = 3000,
        H_0: float = 2500.0,
        gravity: float = 9.81,
        omega: float = _DEFAULT_OMEGA,
        lat: float = 30,
        period: float = 1500,
        use_asselin: bool = False,
        asselin_value: float = 0.1,
        asselin_step: int = 1,
        nesting: bool = False,
        nest_ratio: float = 3,
        calculate_metrics: bool = False,
        nestpos: Tuple[int, int, int, int] = (150, 150, 30, 30),
        dampening: int = 10,
        plotting: bool = False,
        plot_path: str = "sim_output",
        plot_interval: int = 100,
        exp_name: str = "",
        save_outputs: bool = False,
        save_interval: int = 500,
        **kwargs,
    ):
        super().__init__(
            X=X, DX=DX, Y=Y, DY=DY, DT=DT, nt=nt, H_0=H_0, gravity=gravity,
            omega=omega, lat=lat, nesting=nesting, nest_ratio=nest_ratio,
            use_asselin=use_asselin, asselin_value=asselin_value, asselin_step=asselin_step,
            calculate_metrics=calculate_metrics, nestpos=nestpos, dampening=dampening,
            plotting=plotting, plot_path=plot_path, plot_interval=plot_interval,
            exp_name=exp_name, save_outputs=save_outputs, save_interval=save_interval, **kwargs,
        )
        self.period = period
        self.f = 2 * np.pi * omega * np.sin(np.deg2rad(lat))

    def set_initial_conditions(self) -> None:
        pass

    def run_pre_step(self, tt: int) -> None:
        if tt <= self.period:
            self.h1[1, :] = 0.5 * (1 + np.cos(np.pi + 2 * np.pi * (tt - 1) / self.period))

    def boundary_conditions(self, tt: int = 0) -> None:
        self.u1[:, 1] = 0
        self.u1[:, -2] = 0
        self.v1[1, :] = 0
        self.v1[-2, :] = 0
        if tt > self.period:
            self.h1[1, :] = 0

    def interp_topo(self) -> None:
        super().interp_topo()
        self.H2 = self.H2.data if hasattr(self.H2, "data") else self.H2

    def save_figures(self, tt: int, cmap: str = "viridis") -> None:
        fig = plt.figure(figsize=(13, 6), dpi=300)
        fig.suptitle(f"CFL:{self.CFL:06f}, timestep:{tt}", fontsize=16)
        if self.use_nest:
            ax1 = fig.add_subplot(2, 2, 1, projection="3d")
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3, projection="3d")
            ax4 = fig.add_subplot(2, 2, 4)
            roi = patches.Rectangle(
                (self.roi[1], self.roi[0]), self.roi[3], self.roi[2],
                linewidth=1, edgecolor="r", facecolor="none", zorder=100,
            )
            ax2.add_patch(roi)
            for ax in (ax3, ax4):
                ax.set_xlabel("y (km)")
                ax.set_ylabel("x (km)")
            ax3.set_zlabel("z (m)")
        else:
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2 = fig.add_subplot(1, 2, 2)
        vmin, vmax = -1, 1
        h_plot = self.h1 if tt == 0 else self.h2
        ax1.plot_surface(
            self.y_p[1:-1, 1:-1].T, self.x_p[1:-1, 1:-1].T, h_plot[1:-1, 1:-1].T,
            cmap=cmap, edgecolor="none", antialiased=True, vmin=vmin, vmax=vmax,
        )
        img = ax2.pcolormesh(self.y_p.T, self.x_p.T, h_plot.T, vmin=vmin, vmax=vmax, cmap=cmap)
        if self.use_nest:
            h_n = self.h_1 if tt == 0 else self.h_2
            if not hasattr(self, "xnm"):
                self.xnm, self.ynm = np.meshgrid(self.xn, self.yn)
            ax4.pcolormesh(self.ynm.T / 1000, self.xnm.T / 1000, h_n.T, vmin=vmin, vmax=vmax, cmap=cmap)
            ax3.plot_surface(
                self.xnm / 1000, self.ynm / 1000, h_n if tt == 0 else self.h_2,
                cmap=cmap, edgecolor="none", antialiased=True, vmin=vmin, vmax=vmax,
            )
            ax4.set_aspect(1)
        ax2.set_aspect(1)
        ax1.set_xlabel("y (km)")
        ax1.set_ylabel("x (km)")
        ax1.set_zlabel("z (m)")
        ax2.set_xlabel("y (km)")
        ax2.set_ylabel("x (km)")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
        fig.colorbar(img, cax=cbar_ax, label="$h$ (m)")
        plt.savefig(f"{self.path}/{tt:06d}.jpg", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    def plot_speed(self, tt: int) -> None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6), dpi=300)
        fig.suptitle(f"CFL:{self.CFL:06f}, timestep:{tt}", fontsize=16)
        ax1.pcolormesh(self.h2.T, vmin=-1, vmax=1, cmap="twilight_shifted")
        ax1.set_ylabel("x")
        ax1.axhline(100, linestyle="--", color="g", linewidth=0.8)
        ax1.axhline(190, linestyle="--", color="r", linewidth=0.8)
        ax1.axhline(10, linestyle="--", color="b", linewidth=0.8)
        ax1.set_aspect(1)
        ax2.plot(self.h2[:, 100], "g-")
        ax2.plot(self.h2[:, 190], "r-")
        ax2.plot(self.h2[:, 10], "b-")
        ax2.set_ylabel("h (m)")
        ax2.set_ylim(-1.5, 1.5)
        ax2.grid()
        ax3.pcolormesh(self.u2.T, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
        ax3.set_xlabel("y")
        ax3.axhline(100, linestyle="--", color="0.5")
        ax3.set_aspect(1)
        ax4.pcolormesh(self.v2.T, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
        ax4.set_xlabel("y")
        ax4.set_ylabel("x")
        ax4.axhline(100, linestyle="--", color="0.5")
        ax4.set_aspect(1)
        plt.savefig(f"{self.path}/sp_{tt:06d}.jpg", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Custom initial conditions model: user provides h, u, v
# ---------------------------------------------------------------------------

class custom_ic_2dmodel(ShallowWaterModel2DBase):
    """
    Same physics as the base model but initialized from user-provided h, u, v.
    Arrays are copied and reshaped/broadcast to (Y,X), (Y,X+1), (Y+1,X) respectively.
    H_0: float = constant depth (m), or (Y,X) array = depth at each grid point.
    """

    def __init__(
        self,
        h_initial: np.ndarray,
        u_initial: np.ndarray,
        v_initial: np.ndarray,
        X: int = 200,
        DX: float = 1000,
        Y: int = 200,
        DY: float = 1000,
        DT: float = 3.0,
        nt: int = 3000,
        H_0: Union[float, np.ndarray] = 1500.0,
        gravity: float = 9.81,
        omega: float = _DEFAULT_OMEGA,
        lat: float = 30,
        nesting: bool = False,
        nest_ratio: float = 3,
        use_asselin: bool = False,
        asselin_value: float = 0.1,
        asselin_step: int = 1,
        calculate_metrics: bool = False,
        nestpos: Tuple[int, int, int, int] = (150, 150, 30, 30),
        dampening: int = 10,
        plotting: bool = False,
        plot_path: str = "sim_output",
        plot_interval: int = 100,
        exp_name: str = "",
        save_outputs: bool = False,
        save_interval: int = 500,
        **kwargs,
    ):
        super().__init__(
            X=X, DX=DX, Y=Y, DY=DY, DT=DT, nt=nt, H_0=H_0, gravity=gravity,
            omega=omega, lat=lat, nesting=nesting, nest_ratio=nest_ratio,
            use_asselin=use_asselin, asselin_value=asselin_value, asselin_step=asselin_step,
            calculate_metrics=calculate_metrics, nestpos=nestpos, dampening=dampening,
            plotting=plotting, plot_path=plot_path, plot_interval=plot_interval,
            exp_name=exp_name, save_outputs=save_outputs, save_interval=save_interval, **kwargs,
        )
        self._h_initial = np.asarray(h_initial, dtype=float)
        self._u_initial = np.asarray(u_initial, dtype=float)
        self._v_initial = np.asarray(v_initial, dtype=float)

    def set_initial_conditions(self) -> None:
        """Copy user-provided arrays into h0, u0, v0. forward_difference() will compute h1, u1, v1."""
        self.h0 = _ensure_shape(self._h_initial, (self.Y, self.X), "h")
        self.u0 = _ensure_shape(self._u_initial, (self.Y, self.X + 1), "u")
        self.v0 = _ensure_shape(self._v_initial, (self.Y + 1, self.X), "v")


# ---------------------------------------------------------------------------
# Factory: single entry point to create any model
# ---------------------------------------------------------------------------

class ShallowWaterModel:
    """
    Factory to create and run any 2D shallow water model from a single interface.
    Usage:
        model = ShallowWaterModel.create("simple", initialc="e", ...)
        model = ShallowWaterModel.create("channel", period=1500, ...)
        model = ShallowWaterModel.create("custom", h_initial=h, u_initial=u, v_initial=v, X=100, Y=100)
        model.run()
    """

    MODELS = {"simple": simple2dmodel, "channel": channel2dmodel, "custom": custom_ic_2dmodel}

    @classmethod
    def create(cls, model_type: str, **kwargs):
        """
        Create a model instance.
        model_type: "simple" | "channel" | "custom"
        For "custom", pass h_initial, u_initial, v_initial (and grid params X, Y, etc.).
        """
        model_type = model_type.lower().strip()
        if model_type not in cls.MODELS:
            raise ValueError(f"Unknown model type '{model_type}'. Choose from: {list(cls.MODELS.keys())}")
        return cls.MODELS[model_type](**kwargs)

    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """Register an additional model class for create()."""
        cls.MODELS[name] = model_class
