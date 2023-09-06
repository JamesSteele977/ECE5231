import numpy as np
import copy

def get_nd_input(doms: list, 
                 ns: list):
    assert len(doms) == len(ns)
    mgrid_config = [np.s_[dom[0]:dom[1]:n*1j] for dom, n in zip(doms, ns)] 
    grid = np.mgrid[mgrid_config].reshape(len(ns),-1).T
    return grid

test = get_nd_input(doms=[(0, 10),
                          (0, 10),
                          (0, 10)],
                    ns=[11, 11, 11])

np.set_printoptions(threshold=np.inf)
print(test)
