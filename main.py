import numpy as np
import matplotlib.pyplot as plt

def get_nd_input(doms: list, 
                 ns: list):
    assert len(doms) == len(ns)
    mgrid_config = [np.s_[dom[0]:dom[1]:n*1j] for dom, n in zip(doms, ns)] 
    grid = np.mgrid[mgrid_config].reshape(len(ns),-1).T
    return grid

def vec_in(X):
    return [X[...,i] for i in range(X.shape[-1])]

Ep0: float = 8.85418782e-12
E: float = 157e9

F_min: float = 1e-5
F_max: float = 1e-3
C_min: float = 1e-11
C_max: float = 1e-14

s_dom: tuple = (1e-6, 1e-4)
A_dom: tuple = (s_dom[0]**2, s_dom[1]**2)
D_dom: tuple = (1e-7, 1e-4)
L_dom: tuple = (1e-5, 1e-3)

C = lambda w, h, D, F, k: Ep0*((w*h)/(D-(F/k)))
Kax = lambda w, h, L: E*((w*h)/L)
Klv = lambda w, h, L: E*((w*(h**3))/(4*(L**3)))
lCax = lambda w, h, D, F, L: C(w, h, D, F, Kax(w, h, L))
Clv = lambda w, h, D, F, L: C(w, h, D, F, Klv(w, h, L))

paradigms = get_nd_input([s_dom, s_dom, D_dom, (F_min, F_min), L_dom],
                         [7]*5)
vec = vec_in(paradigms)
test_C = np.vectorize(Clv)(*vec)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

thresh = 0.5*np.max(test_C)

test_C[test_C < thresh] = 0

obj = ax.scatter(vec[0]*vec[1], vec[2], vec[-1], c=test_C, cmap='viridis')
plt.colorbar(obj)
plt.show()