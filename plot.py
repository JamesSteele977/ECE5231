import numpy as np
import matplotlib.pyplot as plt

Ep0: float = 8.85418782e-12
E: float = 157e9

F_min: float = 1e-5
F_max: float = 1e-3
C_min: float = 1e-11
C_max: float = 1e-14

s_dom: tuple = (1e-6, 1e-4)
A_dom: tuple = (s_dom[0]^2, s_dom[1]^2)
D_dom: tuple = (1e-7, 1e-4)
L_dom: tuple = (1e-5, 1e-3)

C = lambda w, h, D, F, k: Ep0((w*h)/(D-(F/k)))
Kax = lambda w, h, L: E((w*h)/L)
Klv = lambda w, h, L: E((w*(h**3))/(4*L^3))

Cax = lambda w, h, D, F, L: C(w, h, D, F, Kax(w, h, L))
Clv = lambda w, h, D, F, L: C(w, h, D, F, Klv(w, h, L))

test_paradigms = np.

test_C = np.vectorize(Cax)(test_paradigms)