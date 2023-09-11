import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sens_lib import *

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
        self.optim_weights = np.array([1,1,1], dtype=np.float64)
        self.optim_weights = tf.nn.softmax(self.optim_weights).numpy()
        if self.inputs["optim_config"]["autobound"]:
            self.param_bounds = self._get_autobounds()
        else:
            self.param_bounds = self.inputs["manual_bounds"]
        if self.inputs["optim_config"]["autoinitvals"]:
            self.subclassed_sensor = subclassed_sensor(self._get_autoinitvals())
        else:
            self.subclassed_sensor = subclassed_sensor(self.inputs["manual_init_vals"])
        pass

    def _get_dO_dI(self,
                   sensor_O: tf.Tensor,
                   I_values: tf.Tensor) -> tf.float64:
        return (sensor_O[-1]-sensor_O[0])/(I_values[-1]-I_values[0])

    def _get_nonlinearity(self, 
                          I_values: tf.Tensor, 
                          sensor_O: tf.Tensor, 
                          avg_dO_dI: tf.float64) -> tf.float64:
        error = tf.math.abs(sensor_O-(avg_dO_dI * (I_values-I_values[0]) + sensor_O[0]))
        return tf.reduce_sum(error)/(self.inputs["specs"]["fsi"][1]-self.inputs["specs"]["fsi"][0])

    def _update_loss_weights(self, 
                             footprint,
                             nonlinearity,
                             avg_dO_dI):
        if footprint > self.inputs["specs"]["max_footprint"]:
            if self.footprint-footprint > 0:
                self.optim_weights[0] *= self.inputs["optim_config"]["secondary_lr"]
        if nonlinearity > self.inputs["specs"]["max_nonlinearity"]:
            if self.nonlinearity-nonlinearity > 0:
                self.optim_weights[1] *= self.inputs["optim_config"]["secondary_lr"]
        if avg_dO_dI < self.inputs["specs"]["min_sensitivity"]:
            if self.avg_dO_dI-avg_dO_dI < 0:
                self.optim_weights[2] *= self.inputs["optim_config"]["secondary_lr"]
        self.optim_weights = tf.nn.softmax(self.optim_weights).numpy()
        self.footprint = footprint
        self.nonlinearity = nonlinearity
        self.avg_dO_dI = avg_dO_dI
        pass

    def _get_loss(self) -> tf.float64:
        I_values = tf.cast(
            tf.linspace(self.inputs["specs"]["fsi"][0], 
                        self.inputs["specs"]["fsi"][1],
                        self.inputs["optim_config"]["sample_depth"]),
            dtype=tf.float64
        )
        sensor_O = self.subclassed_sensor._get_output(I_values,
                                                    *self.trainable_variables,
                                                    self.settings["constants"],
                                                    self.inputs["material"])

        avg_dO_dI = self._get_dO_dI(sensor_O, I_values)
        nonlinearity = self._get_nonlinearity(I_values, sensor_O, avg_dO_dI)
        footprint = self.subclassed_sensor._get_footprint(*self.trainable_variables)
        
        loss = self.optim_weights[0]*footprint\
               +self.optim_weights[1]*nonlinearity\
               -self.optim_weights[2]*avg_dO_dI\
        
        return loss, footprint, nonlinearity, avg_dO_dI

    def _bounds_enforcement(self) -> None:
        for i, bound in enumerate(self.param_bounds):
            self.trainable_variables[i].assign(tf.clip_by_value(self.trainable_variables[i],
                                                                bound[0],
                                                                bound[1]))
        pass

    # @tf.function
    def _train_step(self):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            
            loss, footprint, nonlinearity, avg_dO_dI = self._get_loss()

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._bounds_enforcement()
        self._update_loss_weights(footprint, nonlinearity, avg_dO_dI)
        return loss, footprint, nonlinearity, avg_dO_dI

    def _fit(self):
        for name, val in self.subclassed_sensor.__dict__.items():
            self.__setattr__(name, tf.Variable(initial_value=val,
                                               trainable=True,
                                               dtype=tf.float64))
        epochs = self.inputs["optim_config"]["epochs"]
        losses, footprints, nonlinearities, sensitivities =\
            tuple([np.empty((epochs), dtype=np.float32) for i in range(4)])
        params = np.empty((epochs, len(self.trainable_variables)))
        self.loss, self.footprint, self.nonlinearity, self.avg_dO_dI = self._get_loss()
        for epoch in tqdm(range(epochs), desc="Fitting... "):
            losses[epoch], footprints[epoch], nonlinearities[epoch], sensitivities[epoch] =\
                self._train_step()
            params[epoch, :] = self.trainable_variables
        return losses, params, footprints, nonlinearities, sensitivities