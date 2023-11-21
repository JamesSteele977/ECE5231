import numpy as np
from enum import Enum
from dataclasses import dataclass

@dataclass(frozen=True)
class DesignParameters():
    pass

class Constants(Enum):
    pass

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_design: DesignParameters = DesignParameters()