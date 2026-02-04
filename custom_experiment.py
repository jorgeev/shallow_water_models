# %%
import os
import xarray as xr
import numpy as np
from myfirstshallowwatermodel.model2d import ShallowWaterModel
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cmocean.cm as cmo

def create_mask(Yn: int, Xn: int, dampening: int):
    Is = dampening
    mu = 0.5 * (1+np.cos(np.pi*np.arange(0,Is + 1)/Is ))
    hmask = np.zeros((Yn, Xn))

    
    hmask[:Is+1, :] = mu.reshape(Is+1,1)
    hmask[-Is-1:, :] = np.flipud(mu.reshape(Is+1,1))

    
    for ii in range(Is):
        for jj in range(ii+1, Yn-ii):
            hmask[jj, ii] = 0.5 * (1+np.cos(np.pi*ii/Is))
            hmask[jj,-ii] = 0.5 * (1+np.cos(np.pi*ii/Is))
    
    # Inverse the mask
    hmask = 1 - hmask

    return hmask

hmask = create_mask(300, 300, 50)
# Plot the mask
#plt.imshow(hmask, cmap=cmo.balance)
#plt.colorbar()
#plt.show()
# Your own initial h, u, v
ds = xr.open_dataset("initial_conditions.nc")
u = ds.u_geo.data
v = ds.v_geo.data
h = ds.ssh.data
h_o = h.copy()
h = h - np.mean(h)
#u = u - np.mean(u)
#v = v - np.mean(v)
dx = ds.attrs['dx']
dy = ds.attrs['dy']


# %%
#U = np.zeros((500,500))
#U[100:400, 100:400] = u * hmask
#V = np.zeros((500,500))
#V[100:400, 100:400] = v * hmask
#H = np.zeros((500,500))
#H[100:400, 100:400] = h * hmask
U = u * hmask
V = v * hmask
H = h * hmask
#H = gaussian_filter(H, sigma=3)
#U = gaussian_filter(U, sigma=3)
#V = gaussian_filter(V, sigma=3)

#fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#a1 =ax[0].imshow(H, cmap=cmo.balance)
#plt.colorbar(a1)
#a2 = ax[1].imshow(U, cmap=cmo.balance)
#plt.colorbar(a2)
#a3 = ax[2].imshow(V, cmap=cmo.balance)
#plt.colorbar(a3)
#ax[0].set_title("H")
#ax[1].set_title("U")
#ax[2].set_title("V")
#plt.show()

# %%
model = ShallowWaterModel.create("custom", h_initial=H, u_initial=U*0, v_initial=V*0, DX=dx, DY=dy, X=U.shape[1], Y=U.shape[0], nt=1000, DT=5, H_0=5000, use_asselin=True, exp_name="custom_model_debug", plot_interval=10, plotting=True, calculate_metrics=True, lat=35.5)
model.run()

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(model.E_k, label="Kinetic Energy")
ax.plot(model.E_p, label="Potential Energy")
ax.plot(model.E_p + model.E_k, label="Total Energy")
plt.legend()
plt.savefig("custom_model_run_no_asselin_energy.png")

# %%
