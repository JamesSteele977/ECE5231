import numpy as np
import copy

def get_nd_input(domains, 
                 shape: list):
    
    xy = np.mgrid[0:1:3.1j, 0:1:3.1j].reshape(2,-1).T

test = get_nd_input(domains=[(0, 10),
                             (0, 10)],
                    shape=[11, 11])

print(test)
