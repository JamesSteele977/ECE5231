if __name__ == '__main__':

    import os, sys
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    from libs.optim import Optim, OptimConfig, StateVariable
    from libs.sensor import Sensor, SensorConfig
    
    sensor_config: SensorConfig = SensorConfig(
        trainable_variables={'x': [1,3], 'y': [2,4]},
        bandwidth=[1,10],
        input_symbol='Z',
        parameter_relationships=['x > y'],
        footprint='x',
        response='(y + x*3)*Z'
    )
    optim_config: OptimConfig = OptimConfig(
        optimizer='adam',
        epochs=1000,
        bandwidth_sampling_rate=1e1, 
        relationship_sampling_rate=1e1,
        learning_rate=2e-2,
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

    tvars = TestOptim._get_state_variable(StateVariable.TRAINABLE_VARIABLES, all_epochs=True)
    print(tvars.shape)
    plt.plot(tvars[:,0])
    plt.plot(tvars[:,1])
    plt.show()
    