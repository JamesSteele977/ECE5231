if __name__ == '__main__':

    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from libs.optim import Optim, OptimConfig
    from libs.sensor import Sensor, SensorConfig

    sensor_config: SensorConfig = SensorConfig(
        {'x': [1,3], 'y': [2,4]},
        [1,100],
        'I',
        ['x > y'],
        'x * y',
        '(y^2 - x)*I'
    )
    optim_config: OptimConfig = OptimConfig(
        'adam',
        100,
        1e3, 1e3,
        1e-3,
        1, 1, 1
    )

    TestSensor: Sensor = Sensor(sensor_config)
    TestOptim: Optim = Optim(optim_config, TestSensor.sensor_profile)
    TestOptim.__call__()