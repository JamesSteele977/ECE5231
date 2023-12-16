import numpy as np
import matplotlib.pyplot as plt
import json, copy
from tqdm import tqdm

with open("inputs.json", 'r') as f:
    inputs = json.load(f)
with open("settings.json", 'r') as f:
    settings = json.load(f)

E_si = settings["si"]["youngs_mod"]
Ep_not = settings["ep_not"]

def get_mesh(doms: list, 
                 ns: list):
    assert len(doms) == len(ns)
    mgrid_config = [np.s_[dom[0]:dom[1]:n*1j] for dom, n in zip(doms, ns)] 
    grid = np.mgrid[mgrid_config].reshape(len(ns),-1).T
    return grid

def split_mesh(X):
    return [X[...,i] for i in range(X.shape[-1])]

def _get_capacitance(w, h, z, F, k):
    return Ep_not*((w*h)/(z-(F/k)))

def _get_axial_k(w, h, L):
    return E_si*((w*h)/L)

def _get_cantilever_k(w, h, L):
    return E_si*((w*(h**3))/(4*(L**3)))

def get_axial_capacitance(w, h, z, F, L):
    return _get_capacitance(w, h, z, F, _get_axial_k(w, h, L))

def get_cantilever_capacitance(w, h, z, F, L):
    return _get_capacitance(w, h, z, F, _get_cantilever_k(w, h, L))

eq_dict: dict = {
    "capacitive_pressure_axial_single": get_axial_capacitance,
    "capacitive_pressure_cantilever_single": get_cantilever_capacitance,
}

with open("inputs.json", 'r') as f:
    inputs = json.load(f)
with open("settings.json", 'r') as f:
    settings = json.load(f)

phys_dim_range: tuple = (settings["min_dim"], inputs["max_footprint"])
fsi_min_mesh_params = [phys_dim_range]*4
fsi_max_mesh_params = copy.deepcopy(fsi_min_mesh_params)
fsi_min_mesh_params.insert(3, tuple([inputs["fsi"][0]]*2))
fsi_max_mesh_params.insert(3, tuple([inputs["fsi"][1]]*2))

dim_fs = [settings["sample_depth"]]*5
dr_graph = [[phys_dim_range[1]],
            [phys_dim_range[1]],
            [phys_dim_range[1]]]
dr_graph.append([phys_dim_range[0]])

for epoch in tqdm(range(settings["iter_depth"])):

    fsi_min_mesh = get_mesh(fsi_min_mesh_params,
                            dim_fs)

    fsi_max_mesh = get_mesh(fsi_max_mesh_params,
                            dim_fs)

    Cvals_min = np.vectorize(eq_dict[inputs["sensor_type"]])(*split_mesh(fsi_min_mesh))
    Cvals_max = np.vectorize(eq_dict[inputs["sensor_type"]])(*split_mesh(fsi_max_mesh))

    sensitivity_vals = np.abs((Cvals_max-Cvals_min)/(inputs["fsi"][0]-inputs["fsi"][1]))

    cutoff = np.sort(sensitivity_vals, axis=-1)[-int(sensitivity_vals.shape[-1]/1000)]

    superthresh_idxs = np.where(sensitivity_vals >= cutoff)[0]
    superthresh_params = fsi_min_mesh[:,0][superthresh_idxs]
    dr_0 = (np.min(superthresh_params), np.max(superthresh_params))

    superthresh_params = fsi_min_mesh[:,1][superthresh_idxs]
    dr_1 = (np.min(superthresh_params), np.max(superthresh_params))

    superthresh_params = fsi_min_mesh[:,2][superthresh_idxs]
    dr_2 = (np.min(superthresh_params), np.max(superthresh_params))

    superthresh_params = fsi_min_mesh[:,4][superthresh_idxs]
    dr_4 = (np.min(superthresh_params), np.max(superthresh_params))

    fsi_min_mesh_params = [dr_0, dr_1, dr_2, dr_4]
    fsi_max_mesh_params = copy.deepcopy(fsi_min_mesh_params)
    fsi_min_mesh_params.insert(3, tuple([inputs["fsi"][0]]*2))
    fsi_max_mesh_params.insert(3, tuple([inputs["fsi"][1]]*2))

    dr_graph[0].append(dr_0[1])
    dr_graph[1].append(dr_1[1])
    dr_graph[2].append(dr_2[1])
    dr_graph[3].append(dr_4[0])
    

    plt.plot(sensitivity_vals)
    plt.axhline(np.mean(sensitivity_vals))
plt.axhline(inputs["sensitivity"])
plt.show()

plt.clf()
plt.plot(dr_graph[0])
plt.plot(dr_graph[1])
plt.plot(dr_graph[2])
plt.plot(dr_graph[3])
plt.show()

sense_pass = np.where(sensitivity_vals >= inputs["sensitivity"])[0]
sense_win = np.argmax(sensitivity_vals)

print(fsi_min_mesh[sense_pass])
print("BEST:", fsi_min_mesh[sense_win])

    


