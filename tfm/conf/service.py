import sys

import bentoml
from bentoml import Service

from tfm.service.puzzle import Puzzle8MnistService

# This is needed to prevent the following error:
# https://github.com/bentoml/BentoML/issues/3787
del sys.modules["prometheus_client"]

puzzle_runner = bentoml.pytorch.get("8-puzzle:latest").to_runner()

svc = Service(
    "TFM-agranadosb",
    runners=[puzzle_runner],
)

puzzle_service = Puzzle8MnistService(puzzle_runner)
