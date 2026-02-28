import warnings
from typing import Generator, Dict, Any

import numpy as np


# this function is used to permute the arguments for simpler evaluations of the second-derivatives (d^2/dx_i^2,
# without mixing of partial derivatives)
def permuted_cost_function(cost_function, x, permutation):
    xp = x.copy()
    xp[0] = x[permutation]
    xp[permutation] = x[0]
    return cost_function(xp)


def scalar_linesearch_function(x, *args):
    optimize_function = args[0]
    optimise_around = args[1]
    optimise_direction = args[2]
    further_args = args[3:]
    evaluation_point = optimise_around + x * optimise_direction
    assert callable(optimize_function)
    return optimize_function(evaluation_point)


class ValueView():
    def __init__(self, minimiser: Any):
        self._data_delegate = minimiser.estimators
        self._model = minimiser.parameter_names
        self._delegate = minimiser

    def getitem(self, key):
        idx = self._delegate._var2pos[key]
        return self._delegate.estimators[idx]

    def __getitem__(self, key):
        return self.getitem(key)

    def __setitem__(self, key, value):
        index = self._delegate._var2pos[key]
        if isinstance(index, list):
            warnings.warn("Setting multiple values at once is not supported yet!")
        else:
            self._delegate.estimators[index] = value

    def __iter__(self) -> Generator:
        """Get iterator over values."""
        for i in range(len(self)):
            yield self._delegate.estimators[i]

    def __len__(self) -> int:
        """Get number of paramters."""
        return self._delegate.estimators.shape[0]  # type: ignore

    def __eq__(self, other: object) -> bool:
        """Return true if all values are equal."""
        a, b = np.broadcast_arrays(self, other)  # type:ignore
        return bool(np.all(a == b))

    def __repr__(self) -> str:
        """Get detailed text representation."""
        s = f"<{self.__class__.__name__}"
        for k, v in zip(self._delegate._pos2var, self):
            s += f" {k}={v}"
        s += ">"
        return s

    def to_dict(self) -> Dict[str, float]:
        np.float64
        """Obtain dict representation."""
        return {str(k): float(self._delegate.estimators[i]) for i, k in enumerate(self._delegate._pos2var)}
