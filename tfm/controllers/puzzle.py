from typing import Tuple

import torch
from bentoml.io import Image, JSON
from pydantic import BaseModel

from tfm.conf.service import puzzle_service, svc


class InputSchema(BaseModel):
    order: Tuple[int, int, int, int, int, int, int, int, int]


@svc.api(
    input=JSON(pydantic_model=InputSchema),
    output=Image()
)
def puzzle(order):
    return puzzle_service.predict(torch.tensor(order.order))
