import numpy as np
import sympy as sym

""" funcs """
def _get_prime(
    pi: float, 
    pi_diffs: float,
    euler_coeff: float
) -> float:
    return pi-pi_diffs*euler_coeff

def _miller_to_euler(h, k, l):
    normal_vector = np.array([h, k, l], dtype=float)
    normal_vector /= np.linalg.norm(normal_vector)
    theta = np.arccos(normal_vector[2])
    phi = np.arctan2(normal_vector[1], normal_vector[0])
    return phi, theta, 0

def _get_lmn(miller_idx: tuple) -> np.ndarray:
    phi, theta, psi = _miller_to_euler(*miller_idx)
    lmn = np.array(
        (
            (
                np.cos(phi)*np.cos(theta)*np.cos(psi)-np.sin(phi)*np.sin(psi),
                np.sin(phi)*np.cos(theta)*np.cos(psi)+np.cos(phi)*np.sin(psi),
                -np.sin(theta)*np.cos(psi)
            ),
            (
                -np.cos(psi)*np.cos(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi),
                -np.sin(phi)*np.cos(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi),
                np.sin(theta)*np.sin(psi)
            )
        ), 
        dtype=np.float32
    )
    return lmn

def _get_eq(pi: str, euler_coeff) -> str:
    ec = sym.symbols('ec')
    eq = sym.sympify(pi+'-(p_44+p_12-p_11)*ec')
    subs_eq = eq.subs({ec: euler_coeff})
    return sym.N(sym.chop(sym.simplify(subs_eq), threshold=1e-3), 2)

def get_piezo_coeffs(
    si_const: dict,
    miller_idx: tuple
) -> tuple:
    lmn = np.square(_get_lmn(miller_idx))
    euler_coeff_l = lambda x: np.dot(x[0,:], np.roll(x[0,:], 1, -1))*-2
    euler_coeff_t = lambda x: np.dot(x[0,:], x[1,:])
    ecl = euler_coeff_l(lmn)
    ect = euler_coeff_t(lmn)

    pi_diffs = si_const['44']+si_const['12']-si_const['11']

    l_prime = _get_prime(si_const['11'], pi_diffs, ecl)
    t_prime = _get_prime(si_const['12'], pi_diffs, ect)

    leq = _get_eq('p_11', ecl)
    teq = _get_eq('p_12', ect)
    return (l_prime, t_prime, leq, teq)

""" consts """
n_type: dict = {
    'name': 'n-type',
    '11': -102.2e-11,
    '12': 53.4e-11,
    '44': -13.6
}
p_type: dict = {
    'name': 'p-type',
    '11': 6.6e-11,
    '12': -1.1e-11,
    '44': 138.1
}
miller_idxs: list = [
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1)
]

if __name__ == '__main__':
    idx_desc = lambda x: ','.join([str(char) for char in x])

    results: dict = {}
    for si_type in (n_type, p_type):
        for idx in miller_idxs:
            l, t, leq, teq = get_piezo_coeffs(si_type, idx)
            results[f"{si_type['name']} {idx_desc(idx)}"] = {
                'coeffs': (l, t),
                'eq': (leq, teq)
            }

    print('Query\t\t|L\t|T\t\t|Equations')
    print('-'*(8*12))
    for desc, result in results.items():
        print(f"{desc}\t|{np.round(result['coeffs'][0],decimals=2)}\t|{np.round(result['coeffs'][1],decimals=2)}\t\t|L: {result['eq'][0]} T: {result['eq'][1]}")
