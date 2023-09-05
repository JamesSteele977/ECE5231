import numpy as np
import matplotlib.pyplot as plt
from sensor import *
import json

subclassing_dict = {
    "capacitive": {
        "pressure": {
            "axial": AxialCapacitivePressureSensor,
            "cantilever": CatileverCapacitivePressureSensor
        }
    }
}

with open('inputs.json', 'r') as f:
    inputs = json.load(f)
with open('settings.json', 'r') as f:
    settings = json.load(f)


sublcass_dir = inputs["sensor_type"].split('_')
sensor = subclassing_dict[sublcass_dir[0]][sublcass_dir[1]][sublcass_dir[2]](
    settings["fab_constrainst"],
    settings["constants"],
    inputs["material"],
    settings["fitting"],
    inputs["specs"]
)

cloud = sensor._fit()

np.set_printoptions(threshold=np.inf)
print(cloud)