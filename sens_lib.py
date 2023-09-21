import tensorflow as tf

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