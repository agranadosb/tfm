from pydantic import BaseModel


class DatasetConfiguration(BaseModel):
    input_size: int
    batch_size: int
    num_workers: int
    dataset: str


class TrainingConfiguration(BaseModel):
    lr: float
    weight_decay: float
    training: float
    validation: float
    hyp: bool = False


class ModelConfiguration(BaseModel):
    layers: list[int]
    movements_layers: list[int]
    out_channels: int
    input_channels: int
    actions: int
    block: str
    log: bool = True


class Configuration(BaseModel):
    dataset: DatasetConfiguration
    training: TrainingConfiguration
    model: ModelConfiguration
