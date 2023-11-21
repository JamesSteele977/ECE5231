import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple

""" DATA STRUCTURES """
class Constants(Enum):
    MATERIAL: str = '(110) p-type crystalline silicon'
    POISSON_RATIO: float = 0.06
    MEMBRANE_GEOMETRIC_STRESS_CONSTANT: float = 0.3
    PI_ONE_ONE: float = 6.6e-11
    PI_ONE_TWO: float = -1.1e-11
    PI_FOUR_FOUR: float = 138.1e-11

@dataclass(frozen=True)
class DesignParameters():
    appliedVoltage: float
    membraneSideLength: float
    membraneThickness: float

    bandwidth: Tuple[float, float]
    bandwidthSamplingRate: float

""" SECONDARY FUNCTIONS """
def _get_output_voltage(
        applied_voltage: float, 
        ratio_delta_resistance_longitudinal: np.ndarray, 
        ratio_delta_resistance_transverse: np.ndarray
    ) -> np.ndarray:
    return applied_voltage * (
            (ratio_delta_resistance_longitudinal + ratio_delta_resistance_transverse)\
            / (2 + ratio_delta_resistance_longitudinal - ratio_delta_resistance_transverse)
        )

def _get_longitudinal_piezoresistive_coefficient() -> float:
    return 0.5 * (Constants.PI_ONE_ONE.value + Constants.PI_ONE_TWO.value + Constants.PI_FOUR_FOUR.value)

def _get_transverse_pizeoresistive_coefficient() -> float:
    return 0.5 * (Constants.PI_ONE_ONE.value + Constants.PI_ONE_TWO.value - Constants.PI_FOUR_FOUR.value)

def _get_ratio_delta_resistance(
        longitudinal_stress: np.ndarray,
        longitudinal_piezoresistive_coefficient: float,
        transverse_piezoresistive_coefficient: float
    ) -> np.ndarray:
    return longitudinal_stress * (
            longitudinal_piezoresistive_coefficient\
            + Constants.POISSON_RATIO.value * transverse_piezoresistive_coefficient
        )

def _get_longitudinal_stress(
        input_pressure: np.ndarray,
        membrane_side_length: float,
        membrane_thickness: float
    ) -> np.ndarray:
    return input_pressure * Constants.MEMBRANE_GEOMETRIC_STRESS_CONSTANT.value\
        * (membrane_side_length**2 / membrane_thickness**2)

""" PRIMARY FUNCTION """
def design_transfer_function(
        design_parameters: DesignParameters,
    ) -> np.ndarray:
    input_pressure: np.ndarray = np.arange(
        start = design_parameters.bandwidth[0],
        stop = design_parameters.bandwidth[-1],
        step = 1/design_parameters.bandwidthSamplingRate
    )
    longitudinal_stress: np.ndarray = _get_longitudinal_stress(
        input_pressure,
        design_parameters.membraneSideLength,
        design_parameters.membraneThickness
    )
    longitudinal_piezoresistive_coefficient: float = _get_longitudinal_piezoresistive_coefficient()
    transverse_pizeoresistive_coefficient: float = _get_transverse_pizeoresistive_coefficient()
    output_voltage: np.ndarray = _get_output_voltage(
        design_parameters.appliedVoltage,
        _get_ratio_delta_resistance(longitudinal_stress, longitudinal_piezoresistive_coefficient, transverse_pizeoresistive_coefficient),
        _get_ratio_delta_resistance(longitudinal_stress, transverse_pizeoresistive_coefficient, longitudinal_piezoresistive_coefficient)
    )
    return (input_pressure, output_voltage)

def get_sensitivity(pressure: np.ndarray, voltage: np.ndarray) -> float:
    return (voltage[-1] - voltage[0]) / (pressure[-1] - pressure[0])

""" MAIN """
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_design = DesignParameters(
        appliedVoltage = 5,
        membraneSideLength = 500e-6,
        membraneThickness = 5e-6,

        bandwidth = (0.0, 1e3),
        bandwidthSamplingRate = 100
    )

    pressure, voltage = design_transfer_function(test_design)
    sensitivity = get_sensitivity(pressure, voltage)

    plt.plot(pressure, voltage)
    plt.title(f"Piezoresistive Sensor IO Curve | Sensitivity: {sensitivity}")
    plt.ylabel('Volts')
    plt.xlabel('Pascals')
    plt.show()