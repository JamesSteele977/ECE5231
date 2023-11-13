import numpy as np

si = {
    'V': 450e-6*5*10e-6*1e-6,
    'C': 712,
    'p': 2330
}
sin = {
    'V': 450e-6*150e-6*0.25e-6,
    'C': 974,
    'p': 3184
}
sio = {
    'V': 450e-6*150e-6*1e-6,
    'C': 1585,
    'p': 2200
}
ni = {
    'V': 450e-6*5*5e-6*0.5e-6,
    'C': 444,
    'p': 8910
}

cs = [np.product(np.array(tuple(x.values()))).tolist() for x in (si, sin, sio, ni)]
c_total = np.sum(cs)

print(cs, c_total)