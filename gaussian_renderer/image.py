
import types
from typing import Optional, Union
import torch

import numpy as np
from numpy import array as tensor


_Array = Union[np.ndarray, torch.tensor]
np.tensor = np.array


def linear_to_srgb(linear: _Array,
                   eps: Optional[float] = None,
                   xnp: types.ModuleType = torch) -> _Array:
  """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
  if eps is None:
    eps = xnp.tensor(xnp.finfo(xnp.float32).eps)
  srgb0 = 323 / 25 * linear
  srgb1 = (211 * xnp.maximum(eps, linear)**(5 / 12) - 11) / 200
  return xnp.where(linear <= 0.0031308, srgb0, srgb1)
