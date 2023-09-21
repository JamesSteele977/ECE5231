import numpy as np
from sens_opt import *
from sens_lib import *
import json, sys

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

_ = sensor._fit(verbose=True)



