import numpy as np
import tqdm

""" TIER 1 """
class Sensor():
    def __init__(self,
                 fab_constraints: dict,
                 constants: dict,
                 material: str,
                 fitting: dict,
                 specs: dict) -> None:
        self.min_dim = fab_constraints["min_dim"]
        self.epsilon_not = constants["epsilon_not"]
        self.material = material
        self.material_specs = constants["materials"][material]
        self.iter_depth = fitting["iter_depth"]
        self.sample_depth = fitting["sample_depth"]
        self.specs = specs
        pass
    
    def _get_mesh(self,
                  doms: list) -> np.ndarray:
        mgrid_config = [np.s_[dom[0]:dom[1]:self.sample_depth*1j] for dom in doms] 
        return np.mgrid[mgrid_config].reshape(len(doms),-1).T

    def _split_mesh(self,
                    mesh: np.ndarray) -> list:
        return [mesh[...,i] for i in range(mesh.shape[-1])]

    def _fit(self, init_doms):
        doms = init_doms
        for epoch in range(self.iter_depth):

            mesh = self._get_mesh(doms)
        

""" TIER 2 """
class CapacitivePressureSensor(Sensor):
    def __init__(self,
                 fab_constraints: dict,
                 constants: dict,
                 material: str,
                 fitting: dict,
                 specs: dict) -> None:
        super().__init__(fab_constraints,
                         constants,
                         material, 
                         fitting,
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
    
    def _get_axial_capacitance(self, 
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
                 specs: dict) -> None:
        super().__init__(self,
                         fab_constraints,
                         constants,
                         material,
                         fitting,
                         specs)
    
    def _get_axial_k(self, 
                     w: np.float64, 
                     h: np.float64, 
                     L: np.float64):
        return self.material_specs["youngs_mod"]*((w*h)/L)
    
    def _get_cantilever_capacitance(self, 
                                    w: np.float64, 
                                    h: np.float64, 
                                    D: np.float64, 
                                    F: np.float64, 
                                    L: np.float64):
        return self._get_capacitance(w, h, D, F, self._get_cantilever_k(w, h, L))