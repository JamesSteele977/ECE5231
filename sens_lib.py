import tensorflow as tf

""" SENSOR SUPERCLASSES """
class CapacitivePressureSensor():
    def __init__(self, init_vals: tuple) -> None:
        self.w = tf.Variable(initial_value=init_vals[0],
                             trainable=True,
                             dtype=tf.float32)
        self.h = tf.Variable(initial_value=init_vals[1],
                             trainable=True,
                             dtype=tf.float32)
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