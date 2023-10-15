from typing import TypeVar

State = TypeVar('State')
Sample = tuple[State, tuple[State | None, ...]]
