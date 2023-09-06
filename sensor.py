import numpy as np
import tensorflow as tf
from tqdm import tqdm

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
        self.subclassed_sensor = subclassed_sensor(self.inputs["init_vals"])
        pass

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
            footprint_penalty = tf.nn.relu(self.inputs["specs"]["max_footprint"]-footprint, 0)
        
            error = tf.math.abs(sensor_O - (avg_dO_dI * I_values))
            nonlinearity = tf.reduce_sum(error)/(self.inputs["specs"]["fsi"][1]-self.inputs["specs"]["fsi"][0])
            
            loss = -self.inputs["optim_config"]["alpha"]*avg_dO_dI\
                +self.inputs["optim_config"]["beta"]*footprint\
                +self.inputs["optim_config"]["gamma"]*nonlinearity
            
            loss = tf.reduce_mean(tf.where(error > self.inputs["specs"]["max_error_tolerance"],
                                           tf.float32.max, 
                                           loss))
            
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss, footprint, nonlinearity, avg_dO_dI

    def _fit(self):
        for name, val in self.subclassed_sensor.__dict__.items():
            self.__setattr__(name, tf.Variable(initial_value=val,
                                               trainable=True,
                                               dtype=tf.float32))
        losses = np.empty((self.inputs["optim_config"]["epochs"]), dtype=np.float32)
        params = np.empty((self.inputs["optim_config"]["epochs"], len(self.trainable_variables)))
        footprints = np.empty((self.inputs["optim_config"]["epochs"]), dtype=np.float32)
        nonlinearities = np.empty((self.inputs["optim_config"]["epochs"]), dtype=np.float32)
        sensitivities = np.empty((self.inputs["optim_config"]["epochs"]), dtype=np.float32)
        for epoch in tqdm(range(self.inputs["optim_config"]["epochs"]), desc="Fitting... "):
            losses[epoch], footprints[epoch], nonlinearities[epoch], sensitivities[epoch] = self._train_step()
            params[epoch, :] = self.trainable_variables
        return losses, params, footprints, nonlinearities, sensitivities
    
""" SENSOR SUPERCLASSES """
class CapacitivePressureSensor():
    def __init__(self, init_vals: tuple) -> None:
        self.w = tf.Variable(initial_value=init_vals[0],
                             trainable=True,
                             dtype=np.float32)
        self.h = tf.Variable(initial_value=init_vals[1],
                             trainable=True,
                             dtype=np.float32)
        self.z = tf.Variable(initial_value=init_vals[2],
                             trainable=True,
                             dtype=tf.float32)
        self.L = tf.Variable(initial_value=init_vals[3],
                             trainable=True,
                             dtype=tf.float32)
    
    def _get_footprint(self, *args):
        return args[0]*args[3]

    def _get_capacitance(self, 
                         w: tf.float32, 
                         h: tf.float32, 
                         z: tf.float32, 
                         F: tf.float32, 
                         k: tf.float32,
                         constants: dict,
                         *args,
                         **kwargs) -> tf.float32:
        return constants["epsilon_not"]*((w*h)/(z-(F/k)))
    
""" OPERATING LEVEL SENSOR SUBCLASSES """
class AxialCapacitivePressureSensor(CapacitivePressureSensor):
    def __init__(self, init_vals: tuple) -> None:
        super().__init__(init_vals)
        pass

    def _get_axial_k(self, 
                     w: tf.float32, 
                     h: tf.float32, 
                     L: tf.float32,
                     constants: dict,
                     material: str,
                     *wargs,
                     **kwargs) -> tf.float32:
        return constants["materials"][material]["youngs_mod"]*((w*h)/L)
    
    def _get_output(self, 
                    F: tf.float32, 
                    w: tf.float32, 
                    h: tf.float32, 
                    z: tf.float32, 
                    L: tf.float32,
                    constants: dict,
                    material: str,
                    *args,
                    **kwargs) -> tf.float32:
        return self._get_capacitance(w, h, z, F, self._get_axial_k(w, h, L, constants, material), constants)
        
class CatileverCapacitivePressureSensor(CapacitivePressureSensor):
    def __init__(self, init_vals: tuple) -> None:
        super().__init__(init_vals)
        pass
    
    def _get_cantilever_k(self, 
                          w: tf.float32, 
                          h: tf.float32, 
                          L: tf.float32,
                          constants: dict,
                          material: str,
                          *args,
                          **kwargs) -> tf.float32:
        return constants["materials"][material]["youngs_mod"]*((w*(h**3))/(4*(L**3)))
    
    def _get_output(self, 
                    F: tf.float32, 
                    w: tf.float32, 
                    h: tf.float32, 
                    z: tf.float32,  
                    L: tf.float32,
                    constants: dict,
                    material: str,
                    *args,
                    **kwargs) -> tf.float32:
        return self._get_capacitance(w, h, z, F, self._get_cantilever_k(w, h, L, constants, material), constants)