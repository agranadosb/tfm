from typing import TypeVar

from torch import Tensor

State = TypeVar('State')
Sample = tuple[State, tuple[State | None, ...]]

LightsSample = tuple[Tensor, tuple[Tensor, ...]]
TensorSample = tuple[Tensor, tuple[Tensor | None, ...]]
