import os
import xarray as xr
import numpy as np
from myfirstshallowwatermodel.model2d import ShallowWaterModel

# Built-in simple model (cosine / exponential / centered)
model = ShallowWaterModel.create(
    "simple", initialc="e", origin=(50, 50), X=100, Y=100, nt=1000, DT=0.1,
    plotting=True, plot_path="sim_output", plot_interval=100, exp_name="debug_model_"
)
# Output directory (absolute so you know where to find figures)
out_dir = os.path.abspath(model.path)
print(f"Running model: nt={model.nt}, plot_interval={model.plot_interval}")
print(f"Figures will be saved to: {out_dir}")
model.run()
print(f"Done. Check {out_dir} for frames 000000.jpg, 000100.jpg, ...")

# Channel model
model = ShallowWaterModel.create("channel", period=1500, X=200, Y=200, nt=3000, DT=0.1, exp_name="channel_model_", save_interval=100, plotting=True)
model.run()


