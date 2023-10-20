import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def _normalize(X):
    return (X-np.min(X))/(np.ptp(X))

def plot(trained_vars: tuple,
        losses: list,
        footprints: list,
        nonlinearities: list,
        sensitivities: list,
        params: np.ndarray,
        inputs: dict, 
        settings: dict,
        tracked: dict,
        param_names: list,
        _get_output: callable) -> None:
    print("OPTIMIZED PARAMS:")
    for name, var in zip(param_names, trained_vars):
        print(f"{name}: {var}")
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(footprints, 'r--')
    ax[0,0].plot(nonlinearities, 'g--')
    ax[0,0].plot(sensitivities, 'b--')
    ax[0,0].legend(["footprint", "nonlinear", "sensitivity"])
    for i in range(params.shape[-1]):
        ax[0,1].plot(params[:,i])
        ax[0,1].hlines((inputs["bounds"][i]), xmin=0, xmax=params.shape[-1])
    ax[0,1].legend(param_names)
    ax[1,0].plot(_normalize(losses))
    for name, weight in tracked.items():
        ax[1,0].plot(weight)
    fsi = inputs["specs"]["fsi"]
    output = _get_output(fsi,
                        *trained_vars,
                        settings["constants"],
                        inputs["material"])
    ax[1,1].plot(fsi,output)
    plt.show()

""" SENSOR SUPERCLASSES """
class CapacitivePressureSensor():
    def __init__(self, init_vals: tuple) -> None:
        self.w, self.h, self.z, self.L = init_vals
    
    def _get_footprint(self, *args):
        return args[0]*args[3]

    def _get_capacitance(self, 
                         w: tf.float64, 
                         h: tf.float64, 
                         z: tf.float64, 
                         F: tf.float64, 
                         k: tf.float64,
                         constants: dict,
                         *args,
                         **kwargs) -> tf.float64:
        return constants["epsilon_not"]*((w*h)/(z-(F/k)))
    
""" OPERATING LEVEL SENSOR SUBCLASSES """
class AxialCapacitivePressureSensor(CapacitivePressureSensor):
    def __init__(self, init_vals: tuple) -> None:
        super().__init__(init_vals)
        pass

    def _get_axial_k(self, 
                     w: tf.float64, 
                     h: tf.float64, 
                     L: tf.float64,
                     constants: dict,
                     material: str,
                     *wargs,
                     **kwargs) -> tf.float64:
        return constants["materials"][material]["youngs_mod"]*((w*h)/L)
    
    def _get_output(self, 
                    F: tf.float64, 
                    w: tf.float64, 
                    h: tf.float64, 
                    z: tf.float64, 
                    L: tf.float64,
                    constants: dict,
                    material: str,
                    *args,
                    **kwargs) -> tf.float64:
        return self._get_capacitance(w, h, z, F, self._get_axial_k(w, h, L, constants, material), constants)

    def _plot(self, 
            param_names: list = ["width", "height", "z_dist", "length"],
            **kwargs) -> None:
        plot(**kwargs,
                param_names=param_names,
                _get_output=self._get_output)
        pass 

class CatileverCapacitivePressureSensor(CapacitivePressureSensor):
    def __init__(self, init_vals: tuple) -> None:
        super().__init__(init_vals)
        pass
    
    def _get_cantilever_k(self, 
                          w: tf.float64, 
                          h: tf.float64, 
                          L: tf.float64,
                          constants: dict,
                          material: str,
                          *args,
                          **kwargs) -> tf.float64:
        return constants["materials"][material]["youngs_mod"]*((w*(h**3))/(4*(L**3)))
    
    def _get_output(self, 
                    F: tf.float64, 
                    w: tf.float64, 
                    h: tf.float64, 
                    z: tf.float64,  
                    L: tf.float64,
                    constants: dict,
                    material: str,
                    *args,
                    **kwargs) -> tf.float64:
        return self._get_capacitance(w, h, z, F, self._get_cantilever_k(w, h, L, constants, material), constants)

    def _plot(self, 
              param_names: list = ["width", "height", "z_dist", "length"],
              **kwargs) -> None:
        plot(**kwargs,
             param_names=param_names,
             _get_output=self._get_output)
        pass

class TestSensor():
    def __init__(self, init_vals: tuple) -> None:
        self.x, self.y = init_vals
        pass

    def _get_footprint(self,
                       x: tf.float64,
                       y: tf.float64,
                       *args,
                       **kwargs):
        return x*y

    def _get_output(self, 
                    I: tf.float64,
                    x: tf.float64,
                    y: tf.float64,
                    *args,
                    **kwargs) -> tf.float64:
        return (x*I)+y
    
    def _plot(self, 
            param_names: list = ["x", "y"],
            **kwargs) -> None:
        plot(**kwargs,
             param_names=param_names,
             _get_output=self._get_output)
        pass
        



