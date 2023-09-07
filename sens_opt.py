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
            learning_rate=inputs["optim_config"]["learning_rate"],
            # beta_1=0.0,
            # beta_2=0.0
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

    def _bounds_enforcement(self, gradients):
        new_grad = []
        for i, grad in enumerate(gradients):
            bound = self.param_bounds[i]
            if bound != None:
                var = self.trainable_variables[i]
                low_sig = var-bound[0]
                high_sig = bound[1]-var
                limit = np.min((low_sig, high_sig))
                new_grad.append((limit+(limit/tf.math.sigmoid(grad)))/self.optimizer.learning_rate)
            else:
                new_grad.append(grad)
        return tuple(new_grad)

    # @tf.function
    def _train_step(self):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            
            I_values = tf.linspace(self.inputs["specs"]["fsi"][0], 
                                   self.inputs["specs"]["fsi"][1],
                                   self.inputs["optim_config"]["sample_depth"])

            sensor_O = self.subclassed_sensor._get_output(I_values,
                                                          *self.trainable_variables,
                                                          self.settings["constants"],
                                                          self.inputs["material"])

            avg_dO_dI = (sensor_O[-1]-sensor_O[0])/(I_values[-1]-I_values[0])
            
            footprint = self.subclassed_sensor._get_footprint(*self.trainable_variables)
        
            error = tf.math.abs(sensor_O - (avg_dO_dI * I_values))
            nonlinearity = tf.reduce_sum(error)/(self.inputs["specs"]["fsi"][1]-self.inputs["specs"]["fsi"][0])
            
            loss = -self.inputs["optim_config"]["alpha"]*avg_dO_dI\
                +self.inputs["optim_config"]["beta"]*footprint\
                +self.inputs["optim_config"]["gamma"]*nonlinearity
            
            # max_abs_error_bool = tf.logical_not(tf.reduce_max(error) < self.inputs["specs"]["max_abs_error"])
            # max_nonlinearity_bool = tf.logical_not(nonlinearity < self.inputs["specs"]["max_nonlinearity"])
            # min_sensitivity_bool = tf.logical_not(avg_dO_dI > self.inputs["specs"]["min_sensitivity"])
            # footprint_contraints_bool = tf.logical_and(tf.reduce_all(tf.convert_to_tensor(self.trainable_variables)\
            #                                                          > self.inputs["specs"]["max_footprint"]),
            #                                            tf.reduce_all(tf.convert_to_tensor(self.trainable_variables)\
            #                                                          < self.settings["fab_constraints"]["min_dim"]))

            # meta_bool = tf.reduce_sum(
            #     tf.cast((max_abs_error_bool,
            #              max_nonlinearity_bool,
            #              min_sensitivity_bool,
            #              footprint_contraints_bool), 
            #              dtype=np.float32)
            # )
            
            # loss += meta_bool*tf.abs(loss)*2.0

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
        losses = np.empty((epochs), dtype=np.float32)
        params = np.empty((epochs, len(self.trainable_variables)))
        footprints = np.empty((epochs), dtype=np.float32)
        nonlinearities = np.empty((epochs), dtype=np.float32)
        sensitivities = np.empty((epochs), dtype=np.float32)
        for epoch in tqdm(range(epochs), desc="Fitting... "):
            losses[epoch], footprints[epoch], nonlinearities[epoch], sensitivities[epoch] = self._train_step()
            params[epoch, :] = self.trainable_variables
        return losses, params, footprints, nonlinearities, sensitivities