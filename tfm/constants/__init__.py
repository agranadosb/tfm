from tfm.model.base import ConvBlock, ResidualBlock

ORDERED_ORDER = [1, 2, 3, 8, 0, 4, 7, 6, 5]

LEFT_RIGHT_MOVEMENTS = [1, -1]
TOP_BOT_MOVEMENTS = [3, -3]
MOVEMENTS = [*LEFT_RIGHT_MOVEMENTS, *TOP_BOT_MOVEMENTS]

PUZZLE_DATASET = "puzzle8"
LIGHTS_DATASET = "lights-out"

LABEL_TO_STRING = {
    PUZZLE_DATASET: {
        0: "Right",
        1: "Left",
        2: "Bottom",
        3: "Top",
    },
    LIGHTS_DATASET: {i: f"Light {i}" for i in range(25)},
}
MOVEMENT_TO_LABEL = {
    PUZZLE_DATASET: {1: 0, -1: 1, 3: 2, -3: 3},
    LIGHTS_DATASET: {i: i for i in range(25)},
}
LABEL_TO_MOVEMENT = {
    PUZZLE_DATASET: {value: key for key, value in MOVEMENT_TO_LABEL[PUZZLE_DATASET].items()},
    LIGHTS_DATASET: {value: key for key, value in MOVEMENT_TO_LABEL[LIGHTS_DATASET].items()},
}

BLOCKS = {"conv": ConvBlock, "residual": ResidualBlock}
