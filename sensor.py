import numpy as np
import tensorflow as tf

""" TIER 1 """
class Sensor():
    def __init__(self,
                 fab_constraints: dict,
                 constants: dict,
                 material: str,
                 specs: dict,
                 optim_config: dict) -> None:
        self.min_dim = fab_constraints["min_dim"]
        self.epsilon_not = constants["epsilon_not"]
        self.material = material
        self.material_specs = constants["materials"][material]
        self.specs = specs
        self.optim_config = optim_config
        self.optimizer = tf.optimizers.Adam(
            learning_rate=optim_config["learning_rate"]
        )
        pass
    
    def _get_mesh(self,
                  doms: list) -> np.ndarray:
        mgrid_config = [np.s_[dom[0]:dom[1]:self.sample_depth*1j] for dom in doms] 
        return np.mgrid[mgrid_config].reshape(len(doms),-1).T

    def _split_mesh(self,
                    mesh: np.ndarray) -> list:
        return [mesh[...,i] for i in range(mesh.shape[-1])]

    def _fit(self, init_doms):
        w_ = tf.Variable(initial_value=1.0, 
                    trainable=True, 
                    dtype=tf.float32)
        h_ = tf.Variable(initial_value=1.0, 
                            trainable=True, 
                            dtype=tf.float32)
        l_ = tf.Variable(initial_value=1.0, 
                            trainable=True, 
                            dtype=tf.float32)
        for epoch in range(self.optim_config["epochs"]):
            loss = self.train_step()

    @tf.function
    def train_step(self, w_, h_, l_):
        with tf.GradientTape() as tape:
            tape.watch([w_, h_, l_])
            
            # w_ = tf.maximum(w, 1.0)
            # h_ = tf.maximum(h, 1.0)
            # l_ = tf.maximum(l, 1.0)
            
            I_values = tf.linspace(self.specs["fsi"][0], 
                                   self.specs["fsi"][1],
                                   self.sample_depth)

            sensor_O = self.I_values
            avg_dO_dI = tf.reduce_mean(tf.gradients(sensor_O, I_values))
            
            footprint = w_ * l_
        
            linear_O = avg_dO_dI * I_values
            error = tf.math.abs(sensor_O - linear_O)
            linearity = tf.reduce_sum(error)
            
            loss = -self.optim["alpha"]*avg_dO_dI\
                +self.optim["beta"]*footprint\
                -self.optim["gamma"]*linearity

            loss = tf.reduce_mean(tf.where(error > E, tf.float32.max, loss))
            
        grads = tape.gradient(loss, [w_, h_, l_])
        optimizer.apply_gradients(zip(grads, [w_, h_, l_]))
        return loss, w_, h_, l_

    epochs = 10000

    # Training Loop
    losses = np.empty((epochs))
    ws = np.empty((epochs))
    hs = np.empty((epochs))
    ls = np.empty((epochs))
    for epoch in range(epochs):
        loss, w, h, l = train_step()
        # print(f"Epoch {epoch}, Loss: {loss.numpy()}, Design: {w.numpy(), h.numpy(), l.numpy()}")
        losses[epoch] = loss
        ws[epoch] = w
        hs[epoch] = h
        ls[epoch] = l

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)
    plt.yscale('log')
    ax[0].plot(losses)
    plt.yscale('linear')
    ax[0].legend(["loss"])
    ax[1].plot(ws)
    ax[1].plot(hs)
    ax[1].plot(ls)
    ax[1].legend(["width", "height", "length"])
    plt.show()

""" TIER 2 """
class CapacitivePressureSensor(Sensor):
    def __init__(self,
                 fab_constraints: dict,
                 constants: dict,
                 material: str,
                 fitting: dict,
                 specs: dict,
                 optim: dict) -> None:
        super().__init__(fab_constraints,
                         constants,
                         material, 
                         fitting,
                         optim,
                         specs)
        
    def _get_capacitance(self, 
                         w: np.float64, 
                         h: np.float64, 
                         D: np.float64, 
                         F: np.float64, 
                         k: np.float64) -> np.float64:
        return self.epsilon_not*((w*h)/(D-(F/k)))
    
""" TIER 3 """
class AxialCapacitivePressureSensor(CapacitivePressureSensor):
    def __init__(self, 
                 fab_constraints: dict, 
                 constants: dict,
                 material: str, 
                 fitting: dict,
                 optim: dict,
                 specs: dict) -> None:
        super().__init__(fab_constraints, 
                         constants,
                         material,
                         fitting,
                         specs)
        
    def _get_cantilever_k(self, 
                          w: np.float64, 
                          h: np.float64, 
                          L: np.float64):
        return self.material_sepcs["youngs_mod"]*((w*(h**3))/(4*(L**3)))
    
    def _get_output(self, 
                    w: np.float64, 
                    h: np.float64, 
                    D: np.float64, 
                    F: np.float64, 
                    L: np.float64):
        return self._get_capacitance(w, h, D, F, self._get_axial_k(w, h, L))
        
class CatileverCapacitivePressureSensor(CapacitivePressureSensor):
    def __init__(self, 
                 fab_constraints: dict, 
                 constants: dict, 
                 material: str,
                 fitting: dict,
                 optim: dict,
                 specs: dict) -> None:
        super().__init__(self,
                         fab_constraints,
                         constants,
                         material,
                         fitting,
                         optim,
                         specs)
    
    def _get_axial_k(self, 
                     w: np.float64, 
                     h: np.float64, 
                     L: np.float64):
        return self.material_specs["youngs_mod"]*((w*h)/L)
    
    def _get_output(self, 
                    w: np.float64, 
                    h: np.float64, 
                    D: np.float64, 
                    F: np.float64, 
                    L: np.float64):
        return self._get_capacitance(w, h, D, F, self._get_cantilever_k(w, h, L))