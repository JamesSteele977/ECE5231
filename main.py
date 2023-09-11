import numpy as np
import matplotlib.pyplot as plt
from sens_opt import *
from sens_lib import *
import json, sys, pdb

subclassing_dict = {
    "capacitive": {
        "pressure": {
            "axial": AxialCapacitivePressureSensor,
            "cantilever": CatileverCapacitivePressureSensor
        }
    },
    "test":{"":{"":TestSensor}}
}

with open(sys.argv[1], 'r') as f:
    inputs = json.load(f)

sublcass_dir = inputs["sensor_type"].split('_')
subclassed_sensor = subclassing_dict[sublcass_dir[0]][sublcass_dir[1]][sublcass_dir[2]]

sensor = Sensor(settings=inputs["settings"],
                inputs=inputs,
                subclassed_sensor=subclassed_sensor)

losses, params, footprints, nonlinearities, sensitivities = sensor._fit()

# print(f"""
# OPTIMIZED PARAMS:
# Width: {params[-1,0]}
# Height: {params[-1,1]}
# Z Dist: {params[-1,2]}
# Length: {params[-1,3]}
# """)

print(f"""
OPTIMIZED PARAMS:
X: {params[-1,0]}
Y: {params[-1,1]}
""")

def normalize(X):
    return (X-np.min(X))/(np.ptp(X))

# fig, ax = plt.subplots(1, 2)
# ax[0].plot(losses)
# ax[0].legend(["loss"])
# for parameter in range(params.shape[-1]):
#     ax[1].plot(params[:,parameter])
# ax[1].plot(normalize(footprints)*np.max(params), 'r--')
# ax[1].plot(normalize(nonlinearities)*np.max(params), 'g--')
# ax[1].plot(normalize(sensitivities)*np.max(params), 'b--')
# ax[1].axhline(sensor.param_bounds[0][0])
# ax[1].axhline(sensor.param_bounds[0][1])
# ax[1].legend(["width", "height", "z_dist", "length", "footprint", "nonlinear", "dOdI"])
# plt.show()

fig, ax = plt.subplots(1, 2)
ax[0].plot(losses)
ax[0].legend(["loss"])
for parameter in range(params.shape[-1]):
    ax[1].plot(params[:,parameter])
ax[1].plot(normalize(footprints)*np.max(params), 'r--')
ax[1].plot(normalize(nonlinearities)*np.max(params), 'g--')
ax[1].plot(normalize(sensitivities)*np.max(params), 'b--')
ax[1].legend(["x", "y", "footprint", "nonlinear", "dOdI"])
plt.show()