if __name__ == '__main__':

    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from libs.optim import *
    from libs.sensor import Sensor

    sensor_config = {
        "params": {
            "x": [1,2], 
            "y": [0, 4]
        },
        "relations": ["y < x"],
        "expressions": {"z": "x*y"},
        "footprint": "z^2",
        "bandwidth": [0, 100],
        "input_sym": "I",
        "IO": "(x*y+z)*I"
    }
    optim_config = {
        "optimizer": 'Adam',
        "learning_rate": 1e-2,
        "epochs": 50,
        "dI": 1e-3

    }

    TestOptim = Optim(optim_config)
    TestSensor = Sensor(sensor_config)

    solution = TestOptim.fit(TestSensor)