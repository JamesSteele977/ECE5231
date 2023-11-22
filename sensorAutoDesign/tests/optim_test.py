if __name__ == '__main__':

    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    import numpy as np
    import matplotlib.pyplot as plt

    from libs.optim import Optim, OptimConfig, StateVariable
    from libs.sensor import Sensor, SensorConfig

    sensor_config: SensorConfig = SensorConfig(
        {'x': [1,3], 'y': [2,4]},
        [1,10],
        'Z',
        ['x > y'],
        'x * y',
        '(y^2 - x)*Z'
    )
    optim_config: OptimConfig = OptimConfig(
        'adam',
        10,
        1e1, 1e1,
        1e-2,
        1, 1, 1
    )

    TestSensor: Sensor = Sensor(sensor_config)
    TestOptim: Optim = Optim(optim_config, TestSensor.sensor_profile)
    TestOptim.__call__()

    np.set_printoptions(threshold=np.inf)
    for variable_type in StateVariable.__iter__():
        print(variable_type.name, '\n', TestOptim._get_state_variable(variable_type, all_epochs=True), '\n')