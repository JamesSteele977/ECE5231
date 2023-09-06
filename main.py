import numpy as np
import matplotlib.pyplot as plt
from sensor import *
import json, sys

subclassing_dict = {
    "capacitive": {
        "pressure": {
            "axial": AxialCapacitivePressureSensor,
            "cantilever": CatileverCapacitivePressureSensor
        }
    }
}

with open(sys.argv[1], 'r') as f:
    inputs = json.load(f)
with open('settings.json', 'r') as f:
    settings = json.load(f)


sublcass_dir = inputs["sensor_type"].split('_')
subclassed_sensor = subclassing_dict[sublcass_dir[0]][sublcass_dir[1]][sublcass_dir[2]]

sensor = Sensor(settings=settings,
                inputs=inputs,
                subclassed_sensor=subclassed_sensor)

losses, params, footprints, nonlinearities, sensitivities = sensor._fit()

print(f"""OPTIMIZED PARAMS:
Width: {params[-1,0]}
Height: {params[-1,1]}
Z Dist: {params[-1,2]}
Length: {params[-1,3]}""")\

def normalize(X):
    return (X-np.min(X))/(np.ptp(X))

fig, ax = plt.subplots(1, 2)
ax[0].plot(losses)
plt.legend(["loss"])
for parameter in range(params.shape[-1]):
    ax[1].plot(params[:,parameter])
ax[1].plot(normalize(footprints)*np.max(params), 'r--')
ax[1].plot(normalize(nonlinearities)*np.max(params), 'g--')
ax[1].plot(normalize(sensitivities)*np.max(params), 'b--')
plt.legend(["width", "height", "z_dist", "length", "footprint", "nonlinear", "dOdI"])
plt.show()