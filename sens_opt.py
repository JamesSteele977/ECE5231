import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sens_lib import *
import pdb

""" GENERAL SENSOR OPTIMIZATION MODEL """
class Sensor(tf.Module):
    def __init__(self,
                 settings: dict,
                 inputs: dict,
                 subclassed_sensor: object) -> None:
        super().__init__()
        self.settings = settings
        self.inputs = inputs
        self.optimizer = tf.optimizers.Adam(
            learning_rate=inputs["optim_config"]["learning_rate"]
        )
        if self.inputs["optim_config"]["autobound"]:
            self.param_bounds = self._get_autobounds()
        else:
            self.param_bounds = self.inputs["manual_bounds"]
        if self.inputs["optim_config"]["autoinitvals"]:
            self.subclassed_sensor = subclassed_sensor(self._get_autoinitvals())
        else:
            self.subclassed_sensor = subclassed_sensor(self.inputs["manual_init_vals"])
        pass

    def _get_autobounds(self) -> list:
        pass

    def _get_autoinitvals(self) -> list:
        pass

    def _get_dO_dI(self,
                   sensor_O: tf.Tensor,
                   I_values: tf.Tensor) -> tf.float32:
        return (sensor_O[-1]-sensor_O[0])/(I_values[-1]-I_values[0])

    def _get_nonlinearity(self, 
                          I_values: tf.Tensor, 
                          sensor_O: tf.Tensor, 
                          avg_dO_dI: tf.float32) -> tf.float32:
        error = tf.math.abs(sensor_O - (avg_dO_dI * I_values))
        return tf.reduce_sum(error)/(self.inputs["specs"]["fsi"][1]-self.inputs["specs"]["fsi"][0])

    def _get_loss(self) -> tf.float32:
        I_values = tf.linspace(self.inputs["specs"]["fsi"][0], 
                               self.inputs["specs"]["fsi"][1],
                               self.inputs["optim_config"]["sample_depth"])
        sensor_O = self.subclassed_sensor._get_output(I_values,
                                                        *self.trainable_variables,
                                                        self.settings["constants"],
                                                        self.inputs["material"])

        avg_dO_dI = self._get_dO_dI(sensor_O, I_values)
        nonlinearity = self._get_nonlinearity(I_values, sensor_O, avg_dO_dI)
        footprint = self.subclassed_sensor._get_footprint(*self.trainable_variables)
        
        loss = -self.inputs["optim_config"]["alpha"]*avg_dO_dI\
            +self.inputs["optim_config"]["beta"]*footprint\
            +self.inputs["optim_config"]["gamma"]*nonlinearity
        
        return loss, footprint, nonlinearity, avg_dO_dI

    def _bounds_enforcement(self, 
                            gradients: tuple) -> tuple:
        new_grad = []
        for i, grad in enumerate(gradients):
            bound = self.param_bounds[i]
            param = self.trainable_variables[i]
            limit = np.min((param-bound[0], bound[1]-param))
            new_grad.append((limit+(limit/tf.math.sigmoid(grad)))/self.optimizer.learning_rate)
        return tuple(new_grad)

    # @tf.function
    def _train_step(self):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            
            loss, footprint, nonlinearity, avg_dO_dI = self._get_loss()

        grads = tape.gradient(loss, self.trainable_variables)
        grads = self._bounds_enforcement(grads)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss, footprint, nonlinearity, avg_dO_dI

    def _fit(self):
        for name, val in self.subclassed_sensor.__dict__.items():
            self.__setattr__(name, tf.Variable(initial_value=val,
                                               trainable=True,
                                               dtype=tf.float32))
        epochs = self.inputs["optim_config"]["epochs"]
        losses, footprints, nonlinearities, sensitivities =\
            tuple([np.empty((epochs), dtype=np.float32) for i in range(4)])
        params = np.empty((epochs, len(self.trainable_variables)))
        for epoch in tqdm(range(epochs), desc="Fitting... "):
            losses[epoch], footprints[epoch], nonlinearities[epoch], sensitivities[epoch] =\
                self._train_step()
            params[epoch, :] = self.trainable_variables
        return losses, params, footprints, nonlinearities, sensitivities