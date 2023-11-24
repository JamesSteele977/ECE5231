if __name__ == '__main__':

    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    import numpy as np
    import matplotlib.pyplot as plt

    from libs.optim import Optim, OptimConfig, StateVariable
    from libs.sensor import Sensor, SensorConfig
    
    sensor_config: SensorConfig = SensorConfig(
        trainable_variables={'x': [1,3], 'y': [2,4]},
        bandwidth=[1,10],
        input_symbol='Z',
        parameter_relationships=['x > y'],
        footprint='x',
        response='(y + x)*Z'
    )
    optim_config: OptimConfig = OptimConfig(
        optimizer='adam',
        epochs=20,
        bandwidth_sampling_rate=1e1, 
        relationship_sampling_rate=1e1,
        learning_rate=1e-1,
        initial_footprint_loss_weight=1e-1, 
        initial_mean_squared_error_loss_weight=1e-1, 
        initial_sensitivity_loss_weight=1e-1
    )

    TestSensor: Sensor = Sensor(sensor_config)
    TestOptim: Optim = Optim(optim_config, TestSensor.sensor_profile)
    TestOptim.__call__()

    np.set_printoptions(threshold=np.inf)
    for variable_type in StateVariable.__iter__():
        if variable_type != StateVariable.RESPONSE:
            print(variable_type.name, '\n', TestOptim._get_state_variable(variable_type, all_epochs=True), '\n')