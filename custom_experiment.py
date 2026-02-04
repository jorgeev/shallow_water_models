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
    mask = np.zeros((Yn, Xn))

    
    mask[:Is+1, :] = mu.reshape(Is+1,1)
    mask[-Is-1:, :] = np.flipud(mu.reshape(Is+1,1))

    
    for ii in range(Is):
        for jj in range(ii+1, Yn-ii):
            mask[jj, ii] = 0.5 * (1+np.cos(np.pi*ii/Is))
            mask[jj,-ii] = 0.5 * (1+np.cos(np.pi*ii/Is))
    
    # Inverse the mask
    mask = 1 - mask

    return mask



# Initial conditions  h, u, v
ds = xr.open_dataset("initial_conditions.nc")
# print(f"ds: {ds}")
u = ds.u_geo.data
v = ds.v_geo.data
h = ds.ssh.data
print(f"h shape: {h.shape}")
print(f"u shape: {u.shape}")
print(f"v shape: {v.shape}")
h_mask = create_mask(h.shape[0], h.shape[1], 80)
u_mask = create_mask(u.shape[0], u.shape[1], 80)
v_mask = create_mask(v.shape[0], v.shape[1], 80)

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
H = h * h_mask
U = u * u_mask
V = v * v_mask
H = gaussian_filter(H, sigma=1)
#U = gaussian_filter(U, sigma=1)
#V = gaussian_filter(V, sigma=1)



# %%
model = ShallowWaterModel.create("custom", h_initial=H, u_initial=U, v_initial=V, DX=dx, DY=dy, X=H.shape[1], Y=H.shape[0], nt=1000, DT=1, H_0=4500, use_asselin=True, exp_name="custom_model_debug_2", plot_interval=10, plotting=True, calculate_metrics=True, lat=36, boundary="periodic")
model.run()

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(model.E_k, label="Kinetic Energy")
ax.plot(model.E_p, label="Potential Energy")
ax.plot(model.E_p + model.E_k, label="Total Energy")
plt.legend()
plt.savefig("custom_model_run_asselin_energy.png")

# %%
